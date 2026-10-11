# QuantiPhy Agentic Reasoning — baseline

This repository starts with a local video-to-number VLM baseline for the QuantiPhy Challenge. The default Kaggle path runs the public Qwen2.5-VL-3B-Instruct checkpoint directly, so it needs no OpenAI API key and does not fine-tune model weights. The next step is to add specialist agents for visual measurement, physics calculations, and answer checking.

## Contents

- `run_local_baseline.py`: local VLM baseline (Qwen2.5-VL or Qwen3-VL). For each question it samples frames around every time mentioned in the question or prior plus frames covering the whole clip, labels each frame with its timestamp, asks for `Answer: <number> <unit>`, and converts the answer to the unit the question requests. Resumable: predictions are checkpointed after every row.
- `qp_utils.py`: model-free helpers (number/unit parsing and conversion, timestamp extraction, frame selection, prompts). Unit-tested in `tests/`.
- `score_validation.py`: QuantiPhy MRA, per category (S2, D2, S3, D3), macro average, per video source, and seconds per row.
- `kaggle_quickstart.ipynb`: runs several configurations on the validation split and prints a comparison table.
- `run_baseline.py`: optional API baseline (not used for Track B).

## Run on Kaggle

1. New Notebook, **Internet on**, accelerator **GPU T4 x2**.
2. Import `kaggle_quickstart.ipynb` (it clones the `improve-baseline` branch) and run it with `LIMIT = 5` first to check memory and speed, then `LIMIT = 0`.
3. Compare runs in the final table. Each run keeps `outputs/<name>.csv`, `outputs/<name>.json` and `outputs/<name>.config.json`.

## Run from a terminal

```bash
python -m pip install -r requirements.txt
python run_local_baseline.py --csv validation_dataset.csv --video-dir validation_videos \
  --output outputs/qwen3vl8b_direct.csv --model Qwen/Qwen3-VL-8B-Instruct --frames 12 --limit 5
python score_validation.py --predictions outputs/qwen3vl8b_direct.csv --output outputs/qwen3vl8b_direct.json
python -m pytest -q tests   # helper tests, CPU only
```

Useful flags: `--frames` (frame budget, default 12), `--time-window` (seconds around each mentioned time, default 0.1), `--max-side` (resize, default 640), `--prompt-style direct|measure`, `--load-4bit` (8B on a single T4).

## Input and output

The metadata CSV needs `question` and `video_id` (or `video_path`); `ground_truth_prior`, `depth_info` and `fps` are used when present. Validation labels are in `ground_truth_posterior`. The output keeps the input columns and adds `_row_index`, `parsed_value` (converted to the requested unit), `answer_unit`, `raw_response`, `model`, `seconds`. Test labels are hidden; only the official portal scores the test split.

## Kaggle input paths

To inspect the mounted dataset if the paths differ:

```python
from pathlib import Path
for path in Path('/kaggle/input').rglob('*'):
    if path.is_file() and path.suffix.lower() in {'.csv', '.mp4'}:
        print(path)
```

## Next agentic steps

Planned measure-then-compute agent (tools do the measuring, the VLM only plans):

1. Parse the question and prior into a plan: reference object and its quantity, target object and quantity, times, unit.
2. Locate each object by VLM grounding on a frame, then track it with SAM 2 to get a pixel trajectory and pixel size.
3. Get the scale (metres per pixel) from the prior: size, or velocity/acceleration fitted on the reference trajectory; rescale by depth ratio for 3D clips.
4. Compute the target from its pixel measurement and the scale; fall back to the VLM answer when a step fails.


Keep this baseline as the reference. Add one component at a time and compare its validation MRA: for example, a video-measurement agent, a deterministic unit/kinematics solver, or an independent answer-checking agent. Record each prediction CSV and score before changing the next component.
