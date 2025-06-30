from typing import Dict, List, Tuple
from shapely import Polygon

from tracking_core.structures import TrackingData


class SimpleCounter:
    polygon: Polygon

    in_region: Dict[int, TrackingData]
    out_region: Dict[int, TrackingData]
    out_count: int

    def __init__(self, region: List[Tuple[int, int]]):
        self.polygon = Polygon(region)
        self.in_region = {}
        self.out_region = {}
        self.out_count = 0

    def process(self, tracking_data: List[TrackingData]):
        old_in_region = self.in_region
        self.in_region = {}
        self.out_region = {}

        for td in tracking_data:
            if self.polygon.contains(td.centroid):
                self.in_region[td.id] = td
                td.in_zone = True
            else:
                # If not in region then see if it was previously
                # and if it was move it from in to out region
                if td.id in old_in_region:
                    self.out_count += 1
                self.out_region[td.id] = td

    def in_count(self) -> int:
        return len(self.in_region)

    def passed_through_count(self) -> int:
        return self.out_count

    def reset(self):
        self.in_region.clear()
        self.out_region.clear()

    def in_objects(self):
        for v in self.in_region.values():
            yield v

    def passed_through_objects(self):
        for v in self.out_region.values():
            yield v
