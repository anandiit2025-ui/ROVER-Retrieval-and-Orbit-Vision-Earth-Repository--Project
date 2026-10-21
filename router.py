from datetime import date, timedelta

from utils import RoverError


DATASETS = {
    "flood": ["sentinel-1", "sentinel-2"],
    "water": ["sentinel-2", "sentinel-1"],
    "vegetation": ["sentinel-2"],
    "wildfire": ["sentinel-2"],
}


def make_plan(intent, bbox):
    event = intent["event"]
    if event not in DATASETS:
        raise RoverError(f"Unsupported event: {event}")

    start = date.fromisoformat(intent["start_date"])
    end = date.fromisoformat(intent["end_date"])
    if start > end:
        raise RoverError("The start date must come before the end date")

    before_days = 60 if event in {"flood", "water"} else 30
    if start == end:
        raise RoverError("A before/after comparison needs at least two observation dates")
    # Both scenes must come from the interval named in the question.
    # Keep the search near its endpoints and the two windows disjoint.
    first_half_end = start + timedelta(days=((end - start).days - 1) // 2)
    second_half_start = first_half_end + timedelta(days=1)
    before_start = start
    before_end = min(start + timedelta(days=before_days), first_half_end)
    after_start = max(end - timedelta(days=before_days), second_half_start)
    after_end = end
    before_target = start
    after_target = end
    return {
        **intent,
        "bbox": bbox,
        "before_start": before_start.isoformat(),
        "before_end": before_end.isoformat(),
        "after_start": after_start.isoformat(),
        "after_end": after_end.isoformat(),
        "before_target": before_target.isoformat(),
        "after_target": after_target.isoformat(),
        "datasets": DATASETS[event],
    }
