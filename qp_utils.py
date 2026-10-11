"""Model-free helpers for QuantiPhy: numbers, units, timestamps, frames, prompts.

Nothing here imports torch or transformers, so it can be unit-tested on a CPU box.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

PRIOR_COLUMNS = ("ground_truth_prior", "prior", "prior_information", "prior_info")
DEPTH_COLUMNS = ("depth_info", "depth_information", "depth")

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(NUMBER)

# ---------------------------------------------------------------- units
# Each unit maps to (dimension, factor to SI). Dimensions: L, L/T, L/T2, T.
_LENGTH = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "millimetre": 1e-3, "millimetres": 1e-3,
           "cm": 1e-2, "centimeter": 1e-2, "centimeters": 1e-2, "centimetre": 1e-2, "centimetres": 1e-2,
           "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
           "km": 1e3, "kilometer": 1e3, "kilometers": 1e3}
_LEN_ALT = r"mm|cm|km|m|millimet(?:er|re)s?|centimet(?:er|re)s?|kilomet(?:er|re)s?|met(?:er|re)s?"
UNIT_RE = re.compile(
    rf"(?P<len>{_LEN_ALT})\s*(?:/\s*(?P<time>s|sec|second|h|hr|hour)\s*(?P<sq>\^?\s*2|²|squared)?)?(?![a-z])",
    re.IGNORECASE,
)


def parse_unit(text: str) -> tuple[str, float] | None:
    """'cm/s^2' -> ('L/T2', 0.01); 'meters' -> ('L', 1.0); 'km/h' -> ('L/T', 1/3.6)."""
    m = UNIT_RE.match(text.strip())
    if not m:
        return None
    factor = _LENGTH[m.group("len").lower()]
    t = (m.group("time") or "").lower()
    if not t:
        return "L", factor
    tf = 3600.0 if t.startswith("h") else 1.0
    if m.group("sq"):
        return "L/T2", factor / tf**2
    return "L/T", factor / tf


def target_unit(question: str) -> str | None:
    """Unit requested by the question, e.g. 'in cm/s' -> 'cm/s'. None if not stated."""
    m = re.search(rf"\bin\s+((?:{_LEN_ALT})(?:\s*/\s*s(?:\s*\^?\s*2|²)?)?)(?![a-z])", question, re.IGNORECASE)
    return m.group(1) if m else None


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """Convert between compatible units; unchanged when either side is unknown or mismatched."""
    if not (from_unit and to_unit and math.isfinite(value)):
        return value
    a, b = parse_unit(from_unit), parse_unit(to_unit)
    if not a or not b or a[0] != b[0]:
        return value
    return value * a[1] / b[1]


# ---------------------------------------------------------------- answer parsing
_ANSWER_SPLIT = re.compile(r"(?:final\s+)?answer\s*[:=]\s*", re.IGNORECASE)
_VALUE_UNIT_RE = re.compile(rf"({NUMBER})\s*(?:×\s*10\^?\s*({NUMBER}))?\s*([a-zA-Zµ/\^²0-9 ]{{0,16}})")


def parse_answer(text: str, want_unit: str | None = None) -> tuple[float, str | None]:
    """Return (value converted to want_unit when possible, unit the model wrote)."""
    if not text:
        return math.nan, None
    parts = _ANSWER_SPLIT.split(text)
    candidate = parts[-1] if len(parts) > 1 else text
    found = list(_VALUE_UNIT_RE.finditer(candidate))
    if not found:
        return math.nan, None
    m = found[0] if len(parts) > 1 else found[-1]
    try:
        value = float(m.group(1))
        if m.group(2):
            value *= 10 ** float(m.group(2))
    except ValueError:
        return math.nan, None
    unit = None
    tail = m.group(3).strip()
    if tail and parse_unit(tail):
        unit = UNIT_RE.match(tail).group(0)
    if not math.isfinite(value):
        return math.nan, unit
    return convert(abs(value), unit, want_unit), unit


# ---------------------------------------------------------------- timestamps
_TIME_RE = re.compile(rf"(?:\bt\s*=\s*({NUMBER})\s*s?)|(?<![/\w.])({NUMBER})\s*s(?:ec(?:onds?)?)?\b(?!\s*/)")


def mentioned_times(*texts: str) -> list[float]:
    """Times (seconds) mentioned in the question / prior / depth text, in order, deduplicated."""
    out: list[float] = []
    for text in texts:
        if not text:
            continue
        for m in _TIME_RE.finditer(text):
            v = float(m.group(1) or m.group(2))
            if v >= 0 and not any(abs(v - x) < 1e-6 for x in out):
                out.append(v)
    return out


def select_frame_indices(total: int, fps: float, budget: int, key_times: list[float],
                         window: float = 0.1) -> list[int]:
    """Frames at each key time t and t +- window, then evenly spaced frames up to the budget."""
    if total < 1:
        return []
    chosen: list[int] = []

    def add(idx: int) -> None:
        idx = int(min(max(idx, 0), total - 1))
        if idx not in chosen:
            chosen.append(idx)

    step = max(window * fps, 1.0)
    for t in key_times:
        if len(chosen) + 3 > budget:
            break
        c = round(t * fps)
        if c > total - 1 + step:      # time outside the clip: ignore
            continue
        for d in (-step, 0, step):
            add(round(c + d))
    # Fill the rest by farthest-point sampling over the clip, so the extra frames always
    # cover the whole video (start and end included) and avoid the key-time frames.
    for anchor in (0, total - 1):
        if len(chosen) < budget:
            add(anchor)
    candidates = np.arange(total)
    while len(chosen) < min(budget, total):
        gaps = np.min(np.abs(candidates[:, None] - np.array(chosen)[None, :]), axis=1)
        add(int(candidates[int(np.argmax(gaps))]))
    return sorted(chosen)


# ---------------------------------------------------------------- data access
def first_nonempty(row: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name) if hasattr(row, "get") else None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if str(value).strip():
            return str(value).strip()
    return None


def resolve_video_path(row: Any, video_dir: Path) -> Path:
    for column in ("video_path", "video_file", "filepath"):
        value = first_nonempty(row, (column,))
        if value:
            p = Path(value)
            for c in (p, video_dir / p):
                if c.is_file():
                    return c
    video_id = first_nonempty(row, ("video_id",))
    if not video_id:
        raise ValueError("Input CSV needs a video_id column or a valid video_path column")
    name = video_id if Path(video_id).suffix else f"{video_id}.mp4"
    path = video_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def read_frames(video_path: Path, budget: int, key_times: list[float], fps_hint: Any = None,
                window: float = 0.1):
    """Return (frames as [(t, PIL.Image)], fps, duration)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        try:
            hint = float(fps_hint)
            if math.isfinite(hint) and hint > 0:
                fps = hint
        except (TypeError, ValueError):
            pass
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        if total < 1:
            raise ValueError(f"Could not read frames from {video_path}")
        frames = []
        for idx in select_frame_indices(total, fps, budget, key_times, window):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append((idx / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        if not frames:
            raise ValueError(f"No decodable frames in {video_path}")
        return frames, fps, total / fps
    finally:
        cap.release()


# ---------------------------------------------------------------- prompts
def make_prompt(row: Any, fps: float, duration: float, style: str = "direct") -> str:
    prior = first_nonempty(row, PRIOR_COLUMNS)
    depth = first_nonempty(row, DEPTH_COLUMNS)
    question = first_nonempty(row, ("question",))
    if not question:
        raise ValueError("Input CSV has a row with an empty question")
    unit = target_unit(question)
    lines = [
        f"The video is {duration:.2f} s long at {fps:g} fps; each frame above is labelled with its time.",
        "The camera is fixed.",
    ]
    if prior:
        lines.append(f"Known physical prior (treat it as exact, even if it looks unusual): {prior}")
    if depth:
        lines.append(f"Object-to-camera distances: {depth}")
    lines.append(f"Question: {question}")
    if style == "measure":
        lines.append(
            "Work it out by measurement: (1) measure the prior object's quantity in pixels "
            "(size, or displacement per second between the labelled frames); (2) divide the prior's "
            "real value by that pixel value to get the real-world scale; (3) measure the asked "
            "quantity in pixels the same way; (4) multiply by the scale and convert units. "
            "Keep the reasoning short."
        )
    lines.append(
        f"End with exactly one line 'Answer: <number> {unit or '<unit>'}'."
        if style == "measure"
        else f"Reply with only 'Answer: <number> {unit or '<unit>'}'."
    )
    return "\n".join(lines)
