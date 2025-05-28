from dataclasses import dataclass
from typing import Any, Dict, Tuple

from shapely import Point


@dataclass(init=False)
class TrackingData:
    box: Tuple[int, int, int, int]  # xyxy
    id: int
    class_id: int
    centroid: Tuple[int, int]

    def __init__(self, box, id, class_id):
        self.box = box
        self.id = id
        self.class_id = class_id
        self.centroid = Point(
            (int((self.box[0] + self.box[2]) / 2), int((self.box[1] + self.box[3]) / 2))
        )


@dataclass
class TrackingEvent:
    name: str
    attributes: Dict[str, Any]
