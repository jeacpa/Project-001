import json
from typing import List, Tuple

from constants import (
    COUNT_ZONE,
    ZONE_FILE,
)


def save_zone(zone: List[List[int]]) -> None:
    """Save a zone (list of [x, y] points) to the known file."""
    with ZONE_FILE.open("w", encoding="utf-8") as f:
        json.dump(zone, f, indent=2)


def load_zone() -> List[Tuple[int, int]]:
    """
    Load a zone from the known file.
    If no saved zone exists yet, return the default COUNT_ZONE.
    """
    if not ZONE_FILE.exists():
        print("No zone file found, using default")
        return COUNT_ZONE
    try:
        with ZONE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # validate it's a list of [int, int]
        if isinstance(data, list) and all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p)
            for p in data
        ):
            return [tuple(point) for point in data]

    except (OSError, json.JSONDecodeError):
        pass
    # fallback if file is corrupted or invalid
    print("Invalid zone file, using default")
    return COUNT_ZONE
