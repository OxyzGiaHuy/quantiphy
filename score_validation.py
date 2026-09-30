"""Compute the official QuantiPhy MRA metric for validation predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CATEGORIES = ("S2", "D2", "S3", "D3")


def official_category(row: pd.Series) -> str | None:
    video_type = row.get("video_type")
    inference_type = row.get("inference_type")
    if pd.isna(video_type) or pd.isna(inference_type):
        return None
    video_type = str(video_type)
    inference_type = str(inference_type)
    if len(video_type) < 2 or not inference_type:
        return None
    return f"{inference_type[0].upper()}{video_type[1].upper()}"


def mra(prediction: float, target: float) -> float:
    if not math.isfinite(prediction) or not math.isfinite(target) or target == 0:
        return math.nan
    relative_error = abs(abs(prediction) - target) / abs(target)
    return sum(relative_error < (1 - theta) for theta in THRESHOLDS) / len(THRESHOLDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", help="Optional validation CSV; predictions may already contain labels")
    parser.add_argument("--output", default="outputs/validation_score.json")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    if "parsed_value" not in predictions:
        raise SystemExit("Predictions CSV must contain 'parsed_value'")

    truth_col = next((c for c in ("ground_truth_posterior", "ground_truth", "target", "answer") if c in predictions), None)
    if args.ground_truth:
        truth = pd.read_csv(args.ground_truth)
        gt_col = next((c for c in ("ground_truth_posterior", "ground_truth", "target", "answer") if c in truth), None)
        if gt_col is None:
            raise SystemExit("Ground-truth CSV has no ground_truth_posterior column")
        key = next((c for c in ("id", "sample_id") if c in predictions and c in truth), None)
        if key is None:
            key = predictions.columns[0]
            gt_key = truth.columns[0]
        else:
            gt_key = key
        truth_map = truth.set_index(gt_key)[gt_col]
        predictions["_target"] = predictions[key].map(truth_map)
        truth_col = "_target"
    if truth_col is None:
        raise SystemExit("No ground-truth labels found; use the labeled validation split")

    predictions["_prediction"] = pd.to_numeric(predictions["parsed_value"], errors="coerce")
    predictions["_target"] = pd.to_numeric(predictions[truth_col], errors="coerce")
    predictions["_mra"] = [mra(p, y) for p, y in zip(predictions["_prediction"], predictions["_target"])]

    result: dict[str, float | int | None] = {
        "rows": int(len(predictions)),
        "invalid_predictions": int(predictions["_prediction"].isna().sum()),
    }
    category_scores: dict[str, float] = {}
    if "video_type" in predictions and "inference_type" in predictions:
        predictions["_category"] = predictions.apply(official_category, axis=1)
        for category in CATEGORIES:
            values = predictions.loc[predictions["_category"] == category, "_mra"].dropna()
            if not values.empty:
                category_scores[category] = float(values.mean())
                result[f"mra_{category}"] = category_scores[category]
        if all(category in category_scores for category in CATEGORIES):
            result["mra_macro"] = float(np.mean([category_scores[c] for c in CATEGORIES]))
    result["mra_row_mean"] = float(predictions["_mra"].mean())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Score saved to {output_path}")


if __name__ == "__main__":
    main()
