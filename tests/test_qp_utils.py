import math

import pytest

from qp_utils import (convert, make_prompt, mentioned_times, parse_answer, parse_unit,
                      select_frame_indices, target_unit)


@pytest.mark.parametrize("q, unit", [
    ("What is the length of the wood block in cm?", "cm"),
    ("What is the average velocity of the pencil tip moving on the block in cm/s?", "cm/s"),
    ("What is the acceleration of the yellow ball on the left at 2.00s in cm/s^2?", "cm/s^2"),
    ("What is the distance between the two black road signs in meters?", "meters"),
    ("What is the speed of the car at 1s, in m/s?", "m/s"),
    ("What is the orbital diameter of the Io model?", None),
])
def test_target_unit(q, unit):
    assert target_unit(q) == unit


def test_parse_unit_dimensions():
    assert parse_unit("cm/s^2") == ("L/T2", 0.01)
    assert parse_unit("m/s²")[0] == "L/T2"
    assert parse_unit("meters") == ("L", 1.0)
    assert parse_unit("mm") == ("L", 1e-3)
    assert parse_unit("km/h")[0] == "L/T"
    assert parse_unit("kg") is None


def test_convert():
    assert convert(1.5, "m", "cm") == pytest.approx(150)
    assert convert(57.2, "mm", "cm") == pytest.approx(5.72)
    assert convert(36, "km/h", "m/s") == pytest.approx(10)
    assert convert(2.0, "m/s", "cm") == 2.0          # mismatched dimensions: unchanged
    assert convert(2.0, None, "cm") == 2.0


@pytest.mark.parametrize("text, want, value", [
    ("Answer: 9.9 cm", "cm", 9.9),
    ("Answer: 0.099 m", "cm", 9.9),                 # model used another unit: converted
    ("The ball moves ... so v = 1.2 m/s.\nAnswer: 120 cm/s", "cm/s", 120),
    ("Answer: 3.1", "m", 3.1),
    ("about 4 meters", "meters", 4),
    ("Answer: 1.2 × 10^-3 m", "m", 1.2e-3),
    ("Answer: -3.5 m/s^2", "m/s^2", 3.5),           # magnitudes are scored
])
def test_parse_answer(text, want, value):
    got, _ = parse_answer(text, want)
    assert got == pytest.approx(value)


def test_parse_answer_failure():
    assert math.isnan(parse_answer("I cannot tell.", "m")[0])


def test_mentioned_times():
    assert mentioned_times("What is the white ball's average velocity in 1.00s to 2.00s in cm/s?") == [1.0, 2.0]
    assert mentioned_times("What is the speed of the car at 1s, in m/s?") == [1.0]
    assert mentioned_times("t=1.5, ball acceleration = 3.0 m/s^2") == [1.5]
    assert mentioned_times("walking velocity = 1.25 m/s") == []
    assert mentioned_times("speed at time 0.5s", "t=1s, distance = 23.1 m") == [0.5, 1.0]


def test_select_frame_indices_covers_key_times():
    idx = select_frame_indices(total=96, fps=24, budget=12, key_times=[1.0, 2.0], window=0.1)
    assert len(idx) == 12 and idx == sorted(set(idx))
    for t in (1.0, 2.0):
        assert any(abs(i - t * 24) <= 3 for i in idx)
    assert idx[0] == 0 and idx[-1] == 95


def test_select_frame_indices_ignores_out_of_range_and_small_clips():
    assert select_frame_indices(total=5, fps=24, budget=12, key_times=[30.0]) == [0, 1, 2, 3, 4]


def test_make_prompt_mentions_prior_unit_and_depth():
    row = {"question": "What is the length of the boat in meters?", "ground_truth_prior": "speed of the boat = 1.5 m/s",
           "depth_info": "t=0s, distance_boat_camera = 11.8 m"}
    p = make_prompt(row, fps=24, duration=4.0)
    assert "1.5 m/s" in p and "11.8 m" in p and "Answer: <number> meters" in p
    assert "measure" in make_prompt(row, 24, 4.0, style="measure")
