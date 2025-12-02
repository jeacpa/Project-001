import json
import signal
import threading
import time
from typing import List, Optional, Tuple
import cv2

from pydantic import BaseModel, Field

from tracking_core.annotation_util import (
    render_boxes,
    render_info_text,
    render_text,
    render_traffic_light,
    render_zones,
)

from tracking_core.EventManager import NullEventManager
from tracking_core.TrackingManager import TrackingManager
from constants import (
    COUNT_ZONE,
    FRAME_BUFFER_FILE,
    MODEL_NAME,
    TRACKING_CLASSES,
    ZONE_FILE,
)
from webapp.backend.memory_streamer import MemoryStreamer
from pathlib import Path

FRAME_BYTES_PER_PIXEL = 3
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

VIDEO_FILE = "Cashmere.MP4"
TIME_NEAR_END_MS = 450000  # Time in milliseconds near the end of the video
TIME_GOOD_START_MS = 27000
TIME_INTERESTING_START_MS = 55000
TIME_START_MS = TIME_GOOD_START_MS  # Start time in milliseconds for the video

current_frame: Optional[cv2.Mat] = None
frame_lock = threading.Lock()


class ControlRequest(BaseModel):
    action: str
    x: Optional[int] = None
    y: Optional[int] = None
    count_zone: Optional[List[List[int]]] = Field(alias="countZone", default=None)

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        by_alias = True


class ControlState(BaseModel):
    show_zones: bool = Field(alias="showZones", default=False)
    show_light: bool = Field(alias="showLight", default=False)
    show_text: bool = Field(alias="showText", default=False)
    show_boxes: bool = Field(alias="showBoxes", default=False)
    paused: bool = Field(default=False)
    count_zone: List[Tuple[int, int]] = Field(alias="countZone", default=COUNT_ZONE)

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        by_alias = True


class ControlResponse(BaseModel):
    state: ControlState

    class Config:
        allow_population_by_field_name = True


control_state = ControlState(
    show_zones=True,
    show_boxes=True,
    show_light=True,
    show_text=True,
    count_zone=COUNT_ZONE,
)

control_lock = threading.Lock()

stop_event = threading.Event()

tracking_loop_thread: Optional[threading.Thread] = None
tracking_capture: Optional[cv2.VideoCapture] = None
tracking: Optional[TrackingManager] = None

mem_stream = MemoryStreamer(True)

original_sigterm = signal.getsignal(signal.SIGTERM)


def handle_sigterm(signum, frame):
    stop_event.set()

    if callable(original_sigterm):
        original_sigterm(signum, frame)


def handle_sigint(signum, frame):
    stop_event.set()


# Register signal handlers (only in main thread!)
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigterm)


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

def stop_tracking():
    global tracking_capture, tracking

    if tracking_capture and tracking_capture.isOpened():
        tracking_capture.release()

    if tracking:
        tracking.close()


def reset_tracking():
    global tracking_capture, tracking

    stop_tracking()

    tracking_capture = cv2.VideoCapture(VIDEO_FILE)
    tracking_capture.set(cv2.CAP_PROP_POS_MSEC, TIME_START_MS)
    tracking = TrackingManager(
        cap=tracking_capture,
        yolo_model_name=MODEL_NAME,
        tracking_classes=TRACKING_CLASSES,
        count_zone=control_state.count_zone,
        event_manager=NullEventManager(),
        is_live=False,  # Not live, we are reading a video file
        buffer_file_name=FRAME_BUFFER_FILE,
        # no_delay=True
    )

    tracking.advance_frame(False)

def tracking_loop():
    global current_frame, control_state, tracking, mem_stream

    while not stop_event.is_set():

        # If no consumers after some time pause producing frames to preserve CPU/GPU
        # Note: Assumes clocks on consumers are near the same time as producer
        time_since_last_consumer = time.time() - mem_stream.consumer_timestamp()
        if time_since_last_consumer > 5.0:
            print("No consumers detected, pausing frame production")
            time.sleep(1)
            continue

        tracking.advance_frame(control_state.paused)

        if not tracking.current_frame:
            print("End of video reached, restarting")
            reset_tracking()
            continue

        # Write directly to frame for now
        frame_out = tracking.current_frame.raw_frame

        if control_state.show_zones:
            render_zones(tracking.current_frame, tracking.count_zone, frame_out)
        if control_state.show_light:
            render_traffic_light(tracking.current_frame, frame_out)
        if control_state.show_text:
            render_text(tracking.current_frame, tracking.selected_id, frame_out)
        if control_state.show_boxes:
            render_boxes(tracking.current_frame, tracking.selected_id, frame_out)

        render_info_text(tracking.current_frame, frame_out)

        success: bool
        jpeg: Optional[cv2.Mat]
        success, jpeg = cv2.imencode(".jpg", frame_out)
        if not success or jpeg is None:
            continue

        frame_bytes: bytes = jpeg.tobytes()

        mem_stream.produce_frame(frame_bytes)

def run_loop():

    stop_event.clear()

    # Insure zone is loaded
    control_state.count_zone = load_zone()

    print("Initializing tracking...")
    reset_tracking()

    print("Running tracking loop...")

    tracking_loop()

    print("Stopping tracking loop...")
    stop_tracking()

    mem_stream.close()

# @app.websocket("/ws/control")
# async def websocket_control(ws: WebSocket):

#     await ws.accept()
#     try:
#         while True:
#             data = await ws.receive_json()
#             req = ControlRequest.model_validate(data)

#             action = req.action

#             if action == "toggle_zones":
#                 with control_lock:
#                     control_state.show_zones = not control_state.show_zones
#             elif action == "toggle_light":
#                 with control_lock:
#                     control_state.show_light = not control_state.show_light
#             elif action == "toggle_text":
#                 with control_lock:
#                     control_state.show_text = not control_state.show_text
#             elif action == "toggle_boxes":
#                 with control_lock:
#                     control_state.show_boxes = not control_state.show_boxes
#             elif action == "toggle_pause":
#                 with control_lock:
#                     control_state.paused = not control_state.paused
#             elif action == "set_paused":
#                 with control_lock:
#                     control_state.paused = True
#             elif action == "fast_forward" and tracking:
#                 tracking.go_forward(1)
#             elif action == "rewind" and tracking:
#                 tracking.go_back(1)
#             elif action == "restart":
#                 reset_loop()
#             elif action == "infojump":
#                 reset_loop(TIME_INTERESTING_START_MS)
#             elif action == "cursor_pos":
#                 if req.x is not None and req.y is not None:
#                     tracking.set_cursor_pos((req.x, req.y))
#             elif action == "select_box":
#                 tracking.select_box_under_cursor()
#             elif action == "set_zone":
#                 save_zone(req.count_zone)
#                 # Need to reset the loop so that
#                 # a) New zone is loaded
#                 # b) Counters are reset for cars in/out
#                 # c) Frame buffer is cleared because zone has changed
#                 reset_loop()

#             res = ControlResponse(state=control_state)
#             await ws.send_json(jsonable_encoder(res, by_alias=True))

#     except WebSocketDisconnect:
#         print("WS Client disconnected")


run_loop()
