import json
import signal
import threading
import time
from typing import AsyncGenerator, List, Optional, Tuple
import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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

app = FastAPI()


FRAME_BUFFER_COUNT = 3
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

# Enable CORS so your frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def reset_loop(time_start_ms: Optional[int] = TIME_START_MS):
    """
    Resets the video loop
    This function will try to restart the tracking loop thread as well as video capture.
    However, if this function is called from within the tracking loop thread, it will not restart the thread.
    """

    global tracking_loop_thread, tracking_capture, tracking

    in_tracking_loop = (
        tracking_loop_thread is not None
        and threading.current_thread() == tracking_loop_thread
    )

    if not in_tracking_loop:
        if tracking_loop_thread and tracking_loop_thread.is_alive():
            stop_event.set()
            tracking_loop_thread.join()

    if tracking_capture and tracking_capture.isOpened():
        tracking_capture.release()

    if tracking:
        tracking.close()

    stop_event.clear()

    # Insure zone is loaded
    control_state.count_zone = load_zone()

    tracking_capture = cv2.VideoCapture(VIDEO_FILE)
    tracking_capture.set(cv2.CAP_PROP_POS_MSEC, time_start_ms)
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

    if not in_tracking_loop:
        tracking_loop_thread = threading.Thread(target=tracking_loop, daemon=True)
        tracking_loop_thread.start()


@app.get("/api/data")
def get_data():
    return {"message": "Hello from FastAPI"}


@app.on_event("shutdown")
def shutdown_event():
    stop_event.set()


def tracking_loop():
    global current_frame

    while not stop_event.is_set():

        tracking.advance_frame(control_state.paused)

        if not tracking.current_frame:
            print("End of video reached, restarting...")
            reset_loop()
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

        with frame_lock:
            current_frame = frame_out

    stop_event.set()


async def jpeg_stream(request: Request) -> AsyncGenerator[bytes, None]:

    try:
        while not stop_event.is_set():
            if await request.is_disconnected():
                print("Client disconnected")
                break

            frame_copy: Optional[cv2.Mat]
            with frame_lock:
                frame_copy = current_frame.copy() if current_frame is not None else None

            if frame_copy is None:
                time.sleep(0.01)
                continue

            success: bool
            jpeg: Optional[cv2.Mat]
            success, jpeg = cv2.imencode(".jpg", frame_copy)
            if not success or jpeg is None:
                continue

            frame_bytes: bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + f"{len(frame_bytes)}".encode()
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
    finally:
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n" b"Content-Length: 0\r\n\r\n"
        )


@app.get("/video")
async def stream_video(request: Request) -> StreamingResponse:

    print("Streaming")
    return StreamingResponse(
        jpeg_stream(request), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/control")
async def websocket_control(ws: WebSocket):

    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            req = ControlRequest.model_validate(data)

            action = req.action

            if action == "toggle_zones":
                with control_lock:
                    control_state.show_zones = not control_state.show_zones
            elif action == "toggle_light":
                with control_lock:
                    control_state.show_light = not control_state.show_light
            elif action == "toggle_text":
                with control_lock:
                    control_state.show_text = not control_state.show_text
            elif action == "toggle_boxes":
                with control_lock:
                    control_state.show_boxes = not control_state.show_boxes
            elif action == "toggle_pause":
                with control_lock:
                    control_state.paused = not control_state.paused
            elif action == "set_paused":
                with control_lock:
                    control_state.paused = True
            elif action == "fast_forward" and tracking:
                tracking.go_forward(1)
            elif action == "rewind" and tracking:
                tracking.go_back(1)
            elif action == "restart":
                reset_loop()
            elif action == "infojump":
                reset_loop(TIME_INTERESTING_START_MS)
            elif action == "cursor_pos":
                if req.x is not None and req.y is not None:
                    tracking.set_cursor_pos((req.x, req.y))
            elif action == "select_box":
                tracking.select_box_under_cursor()
            elif action == "set_zone":
                save_zone(req.count_zone)
                # Need to reset the loop so that
                # a) New zone is loaded
                # b) Counters are reset for cars in/out
                # c) Frame buffer is cleared because zone has changed
                reset_loop()

            res = ControlResponse(state=control_state)
            await ws.send_json(jsonable_encoder(res, by_alias=True))

    except WebSocketDisconnect:
        print("WS Client disconnected")


reset_loop()
# threading.Thread(target=tracking_loop, daemon=True).start()
