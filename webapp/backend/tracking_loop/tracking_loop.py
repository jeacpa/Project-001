import json
import signal
import threading
import time
from typing import List, Optional, Tuple
import cv2


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
    TIME_GOOD_START_MS,
    TRACKING_CLASSES,
    ZONE_FILE,
)

from webapp.backend.tracking_loop.memory_streamer import MemoryStreamer
import webapp.backend.globals as wb_globals

FRAME_BYTES_PER_PIXEL = 3
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

VIDEO_FILE = "Cashmere.MP4"
TIME_NEAR_END_MS = 450000  # Time in milliseconds near the end of the video
TIME_START_MS = TIME_GOOD_START_MS  # Start time in milliseconds for the video

current_frame: Optional[cv2.Mat] = None
messaging_server_thread: Optional[threading.Thread] = None

tracking_loop_thread: Optional[threading.Thread] = None
tracking_capture: Optional[cv2.VideoCapture] = None

mem_stream = MemoryStreamer(True)


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
    global tracking_capture

    if tracking_capture and tracking_capture.isOpened():
        tracking_capture.release()

    if wb_globals.tracking:
        wb_globals.tracking.close()
        wb_globals.tracking = None

def reset_tracking(start: float = TIME_START_MS):
    global tracking_capture

    stop_tracking()

    tracking_capture = cv2.VideoCapture(VIDEO_FILE)
    tracking_capture.set(cv2.CAP_PROP_POS_MSEC, start)
    wb_globals.tracking = TrackingManager(
        cap=tracking_capture,
        yolo_model_name=MODEL_NAME,
        tracking_classes=TRACKING_CLASSES,
        count_zone=wb_globals.control_state.count_zone,
        event_manager=NullEventManager(),
        is_live=False,  # Not live, we are reading a video file
        buffer_file_name=FRAME_BUFFER_FILE,
        # no_delay=True
    )

    wb_globals.tracking.advance_frame(False)

def tracking_loop():
    global current_frame, mem_stream

    is_consumer_paused = False
    while not wb_globals.stop_event.is_set():

        # If no consumers after some time pause producing frames to preserve CPU/GPU
        # Note: Assumes clocks on consumers are near the same time as producer
        time_since_last_consumer = time.time() - mem_stream.consumer_timestamp()
        if time_since_last_consumer > 5.0:
            if not is_consumer_paused:
                print("No consumers detected, pausing tracking loop...")
                is_consumer_paused = True
            time.sleep(1)
        else:
            if is_consumer_paused:
                print("Consumer detected, resuming tracking loop...")
            is_consumer_paused = False

        if wb_globals.reset_loop_requested != -1:
            reset_tracking(wb_globals.reset_loop_requested)
            wb_globals.reset_loop_requested = -1
            continue

        wb_globals.tracking.advance_frame(wb_globals.control_state.paused)

        if not wb_globals.tracking.current_frame:
            print("End of video reached, restarting")
            reset_tracking()
            continue

        # Write directly to frame for now
        frame_out = wb_globals.tracking.current_frame.raw_frame

        if wb_globals.control_state.show_zones:
            render_zones(wb_globals.tracking.current_frame, wb_globals.tracking.count_zone, frame_out)
        if wb_globals.control_state.show_light:
            render_traffic_light(wb_globals.tracking.current_frame, frame_out)
        if wb_globals.control_state.show_text:
            render_text(wb_globals.tracking.current_frame, wb_globals.tracking.selected_id, frame_out)
        if wb_globals.control_state.show_boxes:
            render_boxes(wb_globals.tracking.current_frame, wb_globals.tracking.selected_id, frame_out)

        render_info_text(wb_globals.tracking.current_frame, frame_out)

        success: bool
        jpeg: Optional[cv2.Mat]
        success, jpeg = cv2.imencode(".jpg", frame_out)
        if not success or jpeg is None:
            continue

        frame_bytes: bytes = jpeg.tobytes()

        mem_stream.produce_frame(frame_bytes)

def run_loop():

    print("Initializing tracking...")
    reset_tracking()

    print("Running tracking loop...")

    tracking_loop()

    print("Stopping tracking loop...")
    stop_tracking()

    mem_stream.close()

