from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Tuple

import numpy as np
from shapely import Point


class LightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


@dataclass(init=False)
class TrackingData:
    box: Tuple[int, int, int, int]  # xyxy
    id: int
    class_id: int
    in_zone: bool

    def __init__(self, box, id, class_id):
        self.box = box
        self.id = id
        self.class_id = class_id
        self.in_zone = False

    @cached_property
    def centroid(self) -> Tuple[int, int]:
        return Point(
            (int((self.box[0] + self.box[2]) / 2), int((self.box[1] + self.box[3]) / 2))
        )


@dataclass
class TrackingFrame:
    frame_index: int
    time_offset: float
    tracking_data: List[TrackingData]  # object id -> TrackingData
    raw_frame: np.ndarray
    zone_clear_time: float
    in_count: int
    passed_through_count: int
    light_color: LightColor
    light_change_time: float
    light_duration: float
    frame_processing_time_ms: float


@dataclass
class TrackingEvent:
    name: str
    attributes: Dict[str, Any]


class VideoReadException(Exception):
    pass


class IntersectionDirection(Enum):
    N = "N"
    S = "S"
    E = "E"
    W = "W"
    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"
