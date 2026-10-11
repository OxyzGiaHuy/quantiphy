"""QuantiPhy baseline with a local open-weight VLM (Qwen2.5-VL or Qwen3-VL); no API key.

Improvements over the first version:
- frames are sampled around every time mentioned in the question / prior (plus coverage
  of the whole clip), each labelled with its timestamp, with fps and duration in the prompt;
- the answer is parsed with its unit and converted to the unit the question asks for;
- works with any Hugging Face image-text-to-text checkpoint, optional 4-bit loading;
- per-row timing and the run configuration are written next to the predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from qp_utils import make_prompt, mentioned_times, parse_answer, read_frames, resolve_video_path, target_unit
from qp_utils import PRIOR_COLUMNS, first_nonempty

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def import_transformers():
    """transformers hides the real cause of a failed lazy import; surface it."""
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        return AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        import importlib
        import transformers
        print(f"transformers {transformers.__version__} import failed: {exc}", flush=True)
        for mod in ("transformers.models.auto.processing_auto", "transformers.models.auto.modeling_auto"):
            try:
                importlib.import_module(mod)
            except Exception as inner:  # the underlying error (e.g. numpy/torch binary mismatch)
                print(f"  root cause in {mod}: {type(inner).__name__}: {inner}", flush=True)
        raise SystemExit("Fix the environment (see README > Kaggle) and restart the session.") from exc


def load_model(model_id: str, load_4bit: bool):
    AutoModelForImageTextToText, AutoProcessor = import_transformers()

    kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        # T4 has no bf16; fp16 on GPU, fp32 on CPU.
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def shrink(image, max_side: int):
    w, h = image.size
    scale = max_side / max(w, h)
    return image if scale >= 1 else image.resize((round(w * scale), round(h * scale)))


def ask(model, processor, frames, prompt: str, max_new_tokens: int, max_side: int) -> str:
    content: list[dict[str, Any]] = []
    for t, image in frames:
        content.append({"type": "text", "text": f"Frame at t={t:.3f} s:"})
        content.append({"type": "image", "image": shrink(image, max_side)})
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="QuantiPhy metadata CSV (validation or test)")
    p.add_argument("--video-dir", required=True, help="Directory containing <video_id>.mp4")
    p.add_argument("--output", default="outputs/validation_predictions.csv")
    p.add_argument("--model", default=os.getenv("VLM_MODEL_ID", MODEL_ID))
    p.add_argument("--frames", type=int, default=12, help="Frame budget per question")
    p.add_argument("--time-window", type=float, default=0.1,
                   help="Extra frames at t +- this many seconds around each mentioned time")
    p.add_argument("--max-side", type=int, default=640, help="Resize frames so the longer side is at most this")
    p.add_argument("--prompt-style", choices=("direct", "measure"), default="direct")
    p.add_argument("--load-4bit", action="store_true", help="bitsandbytes 4-bit (fits 8B on one T4)")
    p.add_argument("--limit", type=int, default=0, help="Only the first N rows (smoke test)")
    p.add_argument("--max-new-tokens", type=int, default=0, help="0 = 32 for direct, 512 for measure")
    args = p.parse_args()

    max_new = args.max_new_tokens or (32 if args.prompt_style == "direct" else 512)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.csv)
    if "question" not in data.columns:
        raise SystemExit("Input CSV must contain a 'question' column")
    if args.limit > 0:
        data = data.head(args.limit).copy()
    data = data.reset_index(drop=False).rename(columns={"index": "_row_index"})
    video_dir = Path(args.video_dir)

    out_path.with_suffix(".config.json").write_text(json.dumps(vars(args) | {"max_new_tokens": max_new}, indent=2))

    # Rows with a numeric prediction are checkpoints, so an interrupted Kaggle session resumes.
    saved: dict[int, dict[str, Any]] = {}
    if out_path.is_file():
        prev = pd.read_csv(out_path)
        if {"_row_index", "parsed_value"} <= set(prev.columns):
            for rec in prev.to_dict(orient="records"):
                if pd.notna(rec.get("parsed_value")):
                    saved[int(rec["_row_index"])] = rec

    print(f"Loading {args.model} ({'4-bit' if args.load_4bit else 'fp16'}) on "
          f"{torch.cuda.device_count()} GPU(s)...", flush=True)
    model, processor = load_model(args.model, args.load_4bit)

    for _, row in data.iterrows():
        i = int(row["_row_index"])
        if i in saved:
            continue
        rec = row.to_dict()
        t0 = time.time()
        try:
            question = str(row["question"])
            times = mentioned_times(question, first_nonempty(row, PRIOR_COLUMNS) or "")
            frames, fps, duration = read_frames(resolve_video_path(row, video_dir), args.frames, times,
                                                row.get("fps"), args.time_window)
            raw = ask(model, processor, frames, make_prompt(row, fps, duration, args.prompt_style),
                      max_new, args.max_side)
            value, unit = parse_answer(raw, target_unit(question))
            rec.update(parsed_value=value, answer_unit=unit, raw_response=raw)
        except Exception as exc:  # keep going; failed rows are retried on the next run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rec.update(parsed_value=math.nan, answer_unit=None, raw_response=f"ERROR: {type(exc).__name__}: {exc}")
        rec.update(model=args.model, seconds=round(time.time() - t0, 2))
        saved[i] = rec
        pd.DataFrame(saved.values()).sort_values("_row_index").to_csv(out_path, index=False)
        print(f"[{i + 1}/{len(data)}] {row.get('video_id')}: {rec['parsed_value']} ({rec['seconds']}s)", flush=True)

    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
