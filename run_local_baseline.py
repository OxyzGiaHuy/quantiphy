"""Run the QuantiPhy baseline with local open-weight Qwen2.5-VL; no API key."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PRIOR_COLUMNS = ("prior", "ground_truth_prior", "prior_information", "prior_info")
DEPTH_COLUMNS = ("depth_info", "depth_information", "depth")


def first_nonempty(row: pd.Series, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return None


def parse_number(text: str) -> float:
    """Parse one scalar; NaN means the generated response was not parseable."""
    if not text:
        return math.nan
    answer_tail = re.split(r"(?:final\s+)?answer\s*:\s*", text, flags=re.IGNORECASE)
    candidate = answer_tail[-1] if len(answer_tail) > 1 else text
    matches = NUMBER_RE.findall(candidate)
    if not matches:
        return math.nan
    try:
        value = float(matches[-1])
        return value if math.isfinite(value) else math.nan
    except ValueError:
        return math.nan


def resolve_video_path(row: pd.Series, video_dir: Path) -> Path:
    for column in ("video_path", "video_file", "filepath"):
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            candidate = Path(str(row[column]).strip())
            if candidate.is_file():
                return candidate
            if not candidate.is_absolute():
                candidate = video_dir / candidate
                if candidate.is_file():
                    return candidate

    if "video_id" not in row.index or pd.isna(row["video_id"]):
        raise ValueError("Input CSV needs a video_id column or a valid video_path column")
    video_id = str(row["video_id"]).strip()
    name = video_id if Path(video_id).suffix else f"{video_id}.mp4"
    path = video_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def read_video_frames(
    video_path: Path,
    frame_count: int,
    fps_from_csv: Any = None,
) -> list[tuple[float, Image.Image]]:
    """Sample evenly spaced RGB frames with their timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps_from_csv is not None and pd.notna(fps_from_csv):
            try:
                provided_fps = float(fps_from_csv)
                if math.isfinite(provided_fps) and provided_fps > 0:
                    fps = provided_fps
            except (TypeError, ValueError):
                pass
        if total_frames < 1:
            raise ValueError(f"Could not read frames from {video_path}")
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        indices = np.linspace(0, total_frames - 1, num=min(frame_count, total_frames), dtype=int)
        frames: list[tuple[float, Image.Image]] = []
        for frame_index in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_index / fps, Image.fromarray(rgb)))
        if not frames:
            raise ValueError(f"No decodable frames in {video_path}")
        return frames
    finally:
        cap.release()


def make_prompt(row: pd.Series) -> str:
    context: list[str] = []
    prior = first_nonempty(row, PRIOR_COLUMNS)
    depth = first_nonempty(row, DEPTH_COLUMNS)
    if prior:
        context.append(f"Provided physical prior: {prior}")
    if depth:
        context.append(f"Provided depth information: {depth}")
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("Input CSV has a row with an empty question")
    prefix = "\n".join(context)
    if prefix:
        prefix += "\n\n"
    return (
        f"{prefix}{question}\n\n"
        "Use the supplied video and physical information. Treat the stated prior as given; "
        "do not replace it with a typical-world assumption. Estimate the requested quantity "
        "and return only one numeric value in the unit requested by the question."
    )


def load_local_model(model_id: str):
    """Load a public Hugging Face checkpoint; Kaggle's GPU is used when available."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def ask_local_model(
    model,
    processor,
    row: pd.Series,
    frames: list[tuple[float, Image.Image]],
    max_new_tokens: int,
) -> tuple[str, float]:
    content: list[dict[str, Any]] = []
    for frame_number, (timestamp, image) in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {frame_number} at t={timestamp:.3f} seconds:"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": make_prompt(row)})
    messages = [{"role": "user", "content": content}]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return raw, parse_number(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="QuantiPhy validation metadata CSV")
    parser.add_argument("--video-dir", required=True, help="Directory containing <video_id>.mp4")
    parser.add_argument("--output", default="outputs/validation_predictions.csv")
    parser.add_argument("--model", default=os.getenv("VLM_MODEL_ID", MODEL_ID))
    parser.add_argument("--frames", type=int, default=6, help="Number of sampled frames per video")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N-row limit for a smoke run")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    input_csv = Path(args.csv)
    video_dir = Path(args.video_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv)
    if "question" not in data.columns:
        raise SystemExit("Input CSV must contain a 'question' column")
    if args.limit > 0:
        data = data.head(args.limit).copy()
    data = data.reset_index(drop=False).rename(columns={"index": "_row_index"})

    # Successful rows are checkpoints so a Kaggle session can be resumed.
    saved: dict[int, dict[str, Any]] = {}
    if output_path.is_file():
        previous = pd.read_csv(output_path)
        if "_row_index" in previous.columns and "parsed_value" in previous.columns:
            for record in previous.to_dict(orient="records"):
                if pd.notna(record.get("parsed_value")):
                    saved[int(record["_row_index"])] = record

    print(f"Loading {args.model} on {'CUDA' if torch.cuda.is_available() else 'CPU'}...")
    model, processor = load_local_model(args.model)
    frame_cache: dict[str, list[tuple[float, Image.Image]]] = {}

    for _, row in data.iterrows():
        row_index = int(row["_row_index"])
        if row_index in saved:
            continue
        video_key = str(row.get("video_id", row.get("video_path", row_index)))
        try:
            if video_key not in frame_cache:
                video_path = resolve_video_path(row, video_dir)
                frame_cache[video_key] = read_video_frames(video_path, args.frames, row.get("fps"))
            raw, parsed = ask_local_model(
                model,
                processor,
                row,
                frame_cache[video_key],
                args.max_new_tokens,
            )
            result = row.to_dict()
            result.update({"parsed_value": parsed, "raw_response": raw, "model": args.model})
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = row.to_dict()
            result.update({
                "parsed_value": math.nan,
                "raw_response": f"ERROR: {type(exc).__name__}: {exc}",
                "model": args.model,
            })

        saved[row_index] = result
        pd.DataFrame(saved.values()).sort_values("_row_index").to_csv(output_path, index=False)
        print(f"[{row_index + 1}/{len(data)}] {video_key}: {result['parsed_value']}")

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
