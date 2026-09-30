# QuantiPhy Agentic Reasoning — baseline

This repository starts with a small video-to-number VLM baseline for the QuantiPhy Challenge. It is intentionally API-based: you can run the first validation experiments without owning a GPU. The design leaves room to add specialist agents (for visual measurement, physics calculations, and answer checks) after establishing a baseline.

## Contents

- `run_baseline.py`: samples frames from each video, sends the frames and question to an OpenAI-compatible vision model, parses a numeric answer, and checkpoints predictions after every row.
- `score_validation.py`: computes the QuantiPhy Mean Relative Accuracy (MRA) metric, including the four-category macro average when category columns are present.
- `kaggle_quickstart.ipynb`: Kaggle notebook template for cloning this repo and running a small validation batch.

## Run on Kaggle

1. Create a Kaggle Notebook and turn **Internet** on.
2. The notebook downloads the public QuantiPhy validation dataset from Hugging Face into `/kaggle/working`; no manual data upload is needed.
3. Add an API key in **Notebook → Add-ons → Secrets** with the name `OPENAI_API_KEY`. This baseline makes one model API request per validation question; API usage may incur charges.
4. Run the cells in `kaggle_quickstart.ipynb`. The notebook uses `validation_dataset.csv` and the accompanying `validation_videos/` folder.

The baseline does not use the notebook GPU. GPU is only needed later if you choose to run a local open-weight VLM or fine-tune a model.

## Run from a terminal

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="..."
python run_baseline.py \
  --csv /path/to/quantiphy_validation.csv \
  --video-dir /path/to/validation_videos \
  --output outputs/validation_predictions.csv \
  --model gpt-5.1 \
  --limit 5
python score_validation.py \
  --predictions outputs/validation_predictions.csv \
  --output outputs/validation_score.json
```

Remove `--limit 5` to process the full CSV. Predictions are saved incrementally, so rerunning the command skips rows that already have a numeric prediction. Rows with missing videos or failed API calls are recorded and can be retried on the next run.

## Input and output

The metadata CSV must contain `question` and either `video_id` or `video_path`. With `video_id`, the script looks for `<video_id>.mp4` under `--video-dir`. It includes `prior` (or `ground_truth_prior`) and `depth_info` in the prompt when those columns are present. The validation set labels answers in the `answer` column. `--frames` controls how many evenly spaced video frames are sent to the model (default: 8).

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
