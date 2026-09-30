# QuantiPhy Agentic Reasoning — baseline

This repository starts with a local video-to-number VLM baseline for the QuantiPhy Challenge. The default Kaggle path runs the public Qwen2.5-VL-3B-Instruct checkpoint directly, so it needs no OpenAI API key and does not fine-tune model weights. The next step is to add specialist agents for visual measurement, physics calculations, and answer checking.

## Contents

- `run_local_baseline.py`: samples frames from each video, runs Qwen2.5-VL-3B locally, parses a numeric answer, and checkpoints predictions after every row.
- `run_baseline.py`: optional API-based baseline; install `requirements-api.txt` and provide an API key only if you choose this route.
- `score_validation.py`: computes the QuantiPhy Mean Relative Accuracy (MRA) metric, including the four-category macro average when category columns are present.
- `kaggle_quickstart.ipynb`: Kaggle notebook template for cloning this repo and running a small validation batch.

## Run on Kaggle

1. Create a Kaggle Notebook, turn **Internet** on, and select a GPU accelerator (T4 is sufficient for this 3B model).
2. Run the cells in `kaggle_quickstart.ipynb`. It downloads the public validation dataset and public Qwen checkpoint from Hugging Face; no secret, API key, or manual data upload is needed.
3. Start with `ROW_LIMIT = 5` to check the setup. Change it to `0` to process the full validation CSV. The model runs locally on Kaggle; this baseline performs inference only and does not train or fine-tune.

The model download is several gigabytes, so keep Kaggle Internet enabled. CPU fallback is supported but will be very slow; use a GPU for practical runs.

## Run from a terminal

```bash
python -m pip install -r requirements.txt
python run_local_baseline.py \
  --csv /path/to/quantiphy_validation.csv \
  --video-dir /path/to/validation_videos \
  --output outputs/validation_predictions.csv \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --limit 5
python score_validation.py \
  --predictions outputs/validation_predictions.csv \
  --output outputs/validation_score.json
```

Remove `--limit 5` (or use `--limit 0`) to process the full CSV. Predictions are saved incrementally, so rerunning the command skips rows that already have a numeric prediction. Rows with missing videos or inference errors are recorded and can be retried on the next run. To use the optional API baseline instead, install `requirements-api.txt`, then run `run_baseline.py` with the API key expected by that script.

## Input and output

The metadata CSV must contain `question` and either `video_id` or `video_path`. With `video_id`, the script looks for `<video_id>.mp4` under `--video-dir`. It includes `prior` (or `ground_truth_prior`) and `depth_info` in the prompt when those columns are present. The validation set labels answers in the `answer` column. `--frames` controls how many evenly spaced video frames are sent to the model (default: 6).

The output retains the input metadata and adds `_row_index`, `parsed_value`, `raw_response`, and `model`. Use the labeled validation split for local MRA scoring. The competition test labels are hidden and are scored only by the official submission portal.

## Kaggle input paths

To inspect the mounted dataset if the paths differ:

```python
from pathlib import Path
for path in Path('/kaggle/input').rglob('*'):
    if path.is_file() and path.suffix.lower() in {'.csv', '.mp4'}:
        print(path)
```

## Next agentic steps

Keep this baseline as the reference. Add one component at a time and compare its validation MRA: for example, a video-measurement agent, a deterministic unit/kinematics solver, or an independent answer-checking agent. Record each prediction CSV and score before changing the next component.
