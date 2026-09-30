"""Small API-based QuantiPhy baseline for local or Kaggle execution."""

from __future__ import annotations

import argparse
import base64
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from openai import OpenAI


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
PRIOR_COLUMNS = ("ground_truth_prior", "prior", "prior_information", "prior_info")
DEPTH_COLUMNS = ("depth_info", "depth_information", "depth")


def first_nonempty(row: pd.Series, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return None


def parse_number(text: str) -> float:
    """Parse a scalar from a concise model response; return NaN on failure."""
    if not text:
        return math.nan
    # Prefer the value after an explicit answer marker, otherwise take the last
    # number because the prompt asks the model to return only its final scalar.
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


def encode_video_frames(
    video_path: Path,
    frame_count: int,
    fps_from_csv: Any = None,
) -> list[dict[str, Any]]:
    """Sample evenly spaced frames and encode them for an OpenAI-compatible VLM."""
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
        content: list[dict[str, Any]] = []
        for frame_index in np.unique(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            timestamp = frame_index / fps
            content.append({"type": "text", "text": f"Video frame at t={timestamp:.3f} seconds:"})
            encoded = base64.b64encode(jpeg.tobytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        if not content:
            raise ValueError(f"No decodable frames in {video_path}")
        return content
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


def ask_model(
    client: OpenAI,
    model: str,
    row: pd.Series,
    frames: list[dict[str, Any]],
    max_retries: int,
) -> tuple[str, float]:
    system = (
        "You are a careful video-based physics measurement assistant. Ground the estimate in "
        "the visible frames and supplied prior/depth information. Check units before answering. "
        "Return a single numeric value only."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": frames + [{"type": "text", "text": make_prompt(row)}]},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            return raw, parse_number(raw)
        except Exception as exc:  # Save per-row failures and continue the Kaggle run.
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 15))
    return f"ERROR: {type(last_error).__name__}: {last_error}", math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="QuantiPhy metadata/validation CSV")
    parser.add_argument("--video-dir", required=True, help="Directory containing <video_id>.mp4")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.1"))
    parser.add_argument("--frames", type=int, default=8, help="Number of frames sampled per video")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N-row limit for a smoke run")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (Kaggle: add it as a Notebook Secret).")
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

    # Keep successful rows as checkpoints so an interrupted Kaggle session can resume.
    saved: dict[int, dict[str, Any]] = {}
    if output_path.is_file():
        previous = pd.read_csv(output_path)
        if "_row_index" in previous.columns and "parsed_value" in previous.columns:
            for record in previous.to_dict(orient="records"):
                if pd.notna(record.get("parsed_value")):
                    saved[int(record["_row_index"])] = record

    client_options: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_options["base_url"] = base_url
    client = OpenAI(**client_options)
    frame_cache: dict[str, list[dict[str, Any]]] = {}

    for _, row in data.iterrows():
        row_index = int(row["_row_index"])
        if row_index in saved:
            continue
        video_key = str(row.get("video_id", row.get("video_path", row_index)))
        try:
            if video_key not in frame_cache:
                video_path = resolve_video_path(row, video_dir)
                frame_cache[video_key] = encode_video_frames(video_path, args.frames, row.get("fps"))
            raw, parsed = ask_model(client, args.model, row, frame_cache[video_key], args.max_retries)
            result = row.to_dict()
            result.update({"parsed_value": parsed, "raw_response": raw, "model": args.model})
        except Exception as exc:
            result = row.to_dict()
            result.update({
                "parsed_value": math.nan,
                "raw_response": f"ERROR: {type(exc).__name__}: {exc}",
                "model": args.model,
            })

        saved[row_index] = result
        pd.DataFrame(saved.values()).sort_values("_row_index").to_csv(output_path, index=False)
        status = result["parsed_value"]
        print(f"[{row_index + 1}/{len(data)}] {video_key}: {status}")

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
