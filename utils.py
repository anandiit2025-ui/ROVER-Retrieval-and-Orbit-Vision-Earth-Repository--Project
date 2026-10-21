import json
import math
from pathlib import Path

from geopy.geocoders import Nominatim


ROOT = Path(__file__).parent


class RoverError(Exception):
    pass


def save_json(filename, data):
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=convert, allow_nan=False)


def geocode(place):
    try:
        result = Nominatim(user_agent="simple-rover-project").geocode(
            place, exactly_one=True, timeout=15
        )
    except Exception as error:
        raise RoverError(f"Could not find the location: {error}") from error

    if result is None:
        raise RoverError(f"Location not found: {place}")

    bounds = result.raw.get("boundingbox")
    if bounds:
        south, north, west, east = map(float, bounds)
        return [west, south, east, north]

    return [
        result.longitude - 0.1,
        result.latitude - 0.1,
        result.longitude + 0.1,
        result.latitude + 0.1,
    ]


def safe_number(value):
    number = float(value)
    return number if math.isfinite(number) else None
