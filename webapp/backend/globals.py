import threading
from typing import Optional
from constants import COUNT_ZONE
from tracking_core.TrackingManager import TrackingManager
from webapp.backend.messaging import ControlState


control_state = ControlState(
    show_zones=True,
    show_boxes=True,
    show_light=True,
    show_text=True,
    count_zone=COUNT_ZONE,
)

stop_event = threading.Event()

tracking: Optional[TrackingManager] = None

# can be set to:
# -1 = no reset requested
# (time offset) = reset loop back to this time
reset_loop_requested: int = -1
