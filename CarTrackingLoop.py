from typing import Optional, Tuple
import cv2

import numpy as np
from ultralytics.solutions.solutions import SolutionAnnotator


import logging

from Clients.sql import SqlClient
from tracking_core.annotation_util import (
    render_boxes,
    render_text,
    render_traffic_light,
    render_zones,
)
from tracking_core.draw_util import inverse_text
from tracking_core.EventManager import EventManager, NullEventManager, SqlEventManager
from tracking_core.TrackingManager import TrackingManager
from constants import (
    COUNT_ZONE,
    FRAME_BUFFER_FILE,
    MODEL_NAME,
    MOUSE_COLOR,
    OUTPUT_VIDEO,
    TEXT_COLOR,
    TRACKING_CLASSES,
    WINDOW_NAME,
)
from tracking_core.structures import TrackingFrame

logging.getLogger("ultralytics").setLevel(logging.ERROR)


# Hack: We want to draw ourselves, thank you very much
def stub_function(*args, **kwargs):
    pass


SolutionAnnotator.draw_region = stub_function
SolutionAnnotator.box_label = stub_function
SolutionAnnotator.display_analytics = stub_function


class Experiment:
    video_path: str
    cap: cv2.VideoCapture
    show_text: bool
    show_boxes: bool
    show_zones: bool
    show_light: bool
    show_mouse: bool
    should_exit: bool
    window_exists: bool
    mouse_pos: Tuple[int, int]
    save_events: bool
    start_offset_ms: int
    half_frames: bool
    sql: Optional[SqlClient]
    paused: bool

    def __init__(
        self,
        video_path: str,
        show_boxes: bool = False,
        show_zones: bool = False,
        show_text: bool = True,
        show_light: bool = False,
        show_mouse: bool = False,
        save_events: bool = False,
        start_offset_ms: int = 0,
        half_frames: bool = False,
    ):
        self.video_path = video_path
        self.show_boxes = show_boxes
        self.show_zones = show_zones
        self.show_light = show_light
        self.show_text = show_text
        self.show_mouse = show_mouse
        self.should_exit = False
        self.window_exists = False
        self.save_events = save_events
        self.start_offset_ms = start_offset_ms
        self.half_frames = half_frames
        self.sql = None
        self.paused = False

    def _render_help(self, frame_image: np.ndarray):
        current_y = 500

        def out(text, draw_ball: bool = False):
            nonlocal current_y
            if draw_ball:
                cv2.circle(frame_image, (20, current_y - 5), 5, (255, 255, 255), -1)
            cv2.putText(
                frame_image,
                text,
                (30, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
            )
            current_y += 30

        cv2.rectangle(frame_image, (10, 470), (280, 750), TEXT_COLOR, cv2.FILLED)
        out("Show Stoplight (T)", self.show_light)
        out("Show Zone (L)", self.show_zones)
        out("Show Mouse (M)", self.show_mouse)
        out("Show Boxes (B)", self.show_boxes)
        out("Half frames (F)", self.half_frames)
        out("Pause (SPC)", self.paused)
        out("Back in time (<-)", self.paused)
        out("Forward in time (->)", self.paused)
        out("Quit (Q)")

    def _render_mouse(self, frame_image: np.ndarray):
        if not hasattr(self, "mouse_pos"):
            return
        cv2.line(
            frame_image,
            (self.mouse_pos[0] - 20, self.mouse_pos[1]),
            (self.mouse_pos[0] + 20, self.mouse_pos[1]),
            MOUSE_COLOR,
            2,
        )
        cv2.line(
            frame_image,
            (self.mouse_pos[0], self.mouse_pos[1] - 20),
            (self.mouse_pos[0], self.mouse_pos[1] + 20),
            MOUSE_COLOR,
            2,
        )

        inverse_text(
            frame_image,
            f"{self.mouse_pos[0]},{self.mouse_pos[1]}",
            (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
            MOUSE_COLOR,
            1,
            0.5,
        )

    def _render_frame(self, frame: TrackingFrame, frame_image: np.ndarray):
        if self.show_zones:
            render_zones(frame, COUNT_ZONE, frame_image)
        if self.show_light:
            render_traffic_light(frame, frame_image)
        if self.show_text:
            render_text(frame, None, frame_image)
        if self.show_boxes:
            render_boxes(frame, None, frame_image)
        if self.show_mouse:
            self._render_mouse(frame_image)

        self._render_help(frame_image)

    def _show_frame(self, frame_image: np.ndarray):
        if not self.window_exists:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1920, 1150)  # Pad for top/bottom toolbars
            cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)

            self.window_exists = True

        cv2.imshow(WINDOW_NAME, frame_image)

    def _check_input(self, tracking_manager: TrackingManager):

        key = cv2.waitKey(1)
        if key == ord("b"):
            self.show_boxes = not self.show_boxes
        elif key == ord("l"):
            self.show_zones = not self.show_zones
        elif key == ord("t"):
            self.show_light = not self.show_light
        elif key == ord("m"):
            self.show_mouse = not self.show_mouse
        elif key == ord("q"):
            self.should_exit = True
        elif key == ord("f"):
            self.half_frames = not self.half_frames
            tracking_manager.set_frame_skipping(self.half_frames)
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == 81:
            tracking_manager.go_back(1)
        elif key == 83:
            tracking_manager.go_forward(1)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            self.should_exit = True

    def _handle_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_offset_ms)

    def _run_video_analysis(self, event_manager: EventManager):

        print("Loading video...")

        self._open_capture()

        tracking = TrackingManager(
            cap=self.cap,
            yolo_model_name=MODEL_NAME,
            tracking_classes=TRACKING_CLASSES,
            count_zone=COUNT_ZONE,
            event_manager=event_manager,
            output_path=OUTPUT_VIDEO,  # No output video for now
            frame_skipping=self.half_frames,
            is_live=False,  # Not live, we are reading a video file
            buffer_file_name=FRAME_BUFFER_FILE,
        )

        while True:
            frame: Optional[TrackingFrame]

            tracking.advance_frame(self.paused)

            frame = tracking.current_frame

            if frame is None or self.should_exit:
                break

            frame_out = np.copy(frame.raw_frame)

            if self.paused:

                self._render_frame(frame, frame_out)
                self._show_frame(frame_out)
                self._check_input(tracking)
                continue

            self._render_frame(frame, frame_out)

            self._show_frame(frame_out)

            tracking.write_frame_to_video(frame, frame_out)

            self._check_input(tracking)

        tracking.close()
        self.cap.release()

        cv2.destroyAllWindows()

    def process(self):

        if self.save_events:
            with SqlClient() as sql:
                self.sql = sql
                self._run_video_analysis(SqlEventManager(sql))
        else:
            self._run_video_analysis(NullEventManager())


# Start 27s in to skip initial cross traffic
exp = Experiment(
    video_path="Cashmere.MP4",
    start_offset_ms=27000,
    # start_offset_ms=70000,
    show_light=True,
    show_boxes=True,
    show_zones=True,
    save_events=False,
)

exp.process()
