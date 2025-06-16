from typing import Optional, Tuple
import cv2

import numpy as np
from ultralytics.solutions.solutions import SolutionAnnotator


import logging

from Clients.sql import SqlClient
from EventManager import EventManager, NullEventManager, SqlEventManager
from TrackingManager import TrackingManager
from constants import (
    BLACK_LIGHT,
    BOX_LINE_THICKNESS,
    BOX_TEXT_COLOR,
    BOX_TEXT_SCALE,
    BOX_TEXT_THICKNESS,
    COUNT_ZONE,
    FRAME_BUFFER_FILE,
    GREEN_LIGHT,
    LINE_COLOR,
    LINE_THICKNESS,
    MODEL_NAME,
    MOUSE_COLOR,
    OUTPUT_VIDEO,
    RED_LIGHT,
    TEXT_COLOR,
    TEXT_LINE_HEIGHT,
    TEXT_SCALE,
    TEXT_THICKNESS,
    TRACKING_CLASSES,
    TRACKING_LABELS,
    WINDOW_NAME,
    YELLOW_LIGHT,
    ZONE_CLEAR_COUNTDOWN_SEC,
)
from structures import LightColor, TrackingData, TrackingFrame

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

    def _interpolate_color(
        self, start_color, end_color, t: float
    ) -> tuple[int, int, int]:
        """Linearly interpolate between two BGR colors"""
        return tuple(
            [
                int(start + (end - start) * t)
                for start, end in zip(start_color, end_color)
            ]
        )

    def _render_zones(self, frame: TrackingFrame, frame_image: np.ndarray):

        line_color = LINE_COLOR

        if frame.zone_clear_time > 0:
            elapsed = (frame.time_offset - frame.zone_clear_time) / 1000
            t = min(elapsed / ZONE_CLEAR_COUNTDOWN_SEC, 1.0)
            line_color = self._interpolate_color(LINE_COLOR, (0, 0, 255), t)
        elif frame.zone_clear_time == -1:
            # Flash every 500ms
            if frame.time_offset % 1000 < 500:
                line_color = (0, 0, 255)
            else:
                return

        cv2.polylines(
            frame_image,
            [np.array(COUNT_ZONE, dtype=np.int32)],
            isClosed=True,
            color=line_color,
            thickness=LINE_THICKNESS,
        )

    def _render_text(self, frame: TrackingFrame, frame_image: np.ndarray):
        current_y = 50

        def out(text):
            nonlocal current_y
            cv2.putText(
                frame_image,
                text,
                (10, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
            )
            current_y += TEXT_LINE_HEIGHT

        out(f"Frame: {frame.frame_index}")
        out(f"Time: {(frame.time_offset/1000):.2f}s")
        out(f"Objects in: {frame.in_count}")
        out(f"Objects out: {frame.passed_through_count}")
        out(f"Think time: {frame.frame_processing_time_ms}ms")

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

    def _center_text(
        self, frame_image: np.ndarray, text, xy_center, color, thickness=1, scale=1.0
    ):
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        cv2.putText(
            frame_image,
            text,
            (xy_center[0] - text_width // 2, xy_center[1] + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _inverse_text(
        self, frame_image: np.ndarray, text, xy, color, thickness=1, scale=1.0
    ):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        baseline += 2
        box_coords1 = (xy[0] - 4, xy[1] - text_height - 4)
        box_coords2 = (xy[0] + text_width + 4, xy[1] + baseline)

        cv2.rectangle(frame_image, box_coords1, box_coords2, color, cv2.FILLED)
        cv2.putText(
            frame_image,
            text,
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _render_traffic_light(self, frame: TrackingFrame, frame_image: np.ndarray):
        x, y, w, h = 1800, 1, 80, 300  # x, y coordinates, width, height
        padding = 7
        circle_radius = (w - 2 * padding) // 2

        cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 0, 0), -1)

        cv2.circle(
            frame_image,
            (x + w // 2, y + h // 4),
            circle_radius,
            RED_LIGHT if frame.light_color == LightColor.RED else BLACK_LIGHT,
            -1,
        )

        cv2.circle(
            frame_image,
            (x + w // 2, y + h // 2),
            circle_radius,
            YELLOW_LIGHT if frame.light_color == LightColor.YELLOW else BLACK_LIGHT,
            -1,
        )
        cv2.circle(
            frame_image,
            (x + w // 2, y + 3 * h // 4),
            circle_radius,
            GREEN_LIGHT if frame.light_color == LightColor.GREEN else BLACK_LIGHT,
            -1,
        )

        if (
            frame.light_change_time
            and frame.light_duration
            and (
                frame.light_color == LightColor.RED
                or frame.light_color == LightColor.GREEN
            )
        ):

            elapsed = round((frame.time_offset - frame.light_change_time) / 1000)
            remaining = max(0, round(frame.light_duration / 1000) - elapsed)

            if frame.light_color == LightColor.RED:
                red_text = f"{elapsed}s"
                green_text = f"{remaining}s"
                red_color = (255, 255, 255)
                green_color = (0, 0, 0)
            else:
                green_text = f"{elapsed}s"
                red_text = f"{remaining}s"
                red_color = (0, 0, 0)
                green_color = (255, 255, 255)

            self._center_text(
                frame_image,
                red_text,
                (x + w // 2, y + h // 4),
                color=green_color,
                scale=0.6,
            )
            self._center_text(
                frame_image,
                green_text,
                (x + w // 2, y + 3 * h // 4),
                color=red_color,
                scale=0.6,
            )

    def _render_single_box(self, frame_image: np.ndarray, td: TrackingData, color):
        x1, y1, x2, y2 = td.box
        label = TRACKING_LABELS[td.class_id]
        label = f"{label}/{td.id}"

        cv2.rectangle(
            frame_image,
            (x1, y1),
            (x2, y2),
            color,
            BOX_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        self._inverse_text(
            frame_image,
            label,
            (x1, y1),
            BOX_TEXT_COLOR,
            BOX_TEXT_THICKNESS,
            BOX_TEXT_SCALE,
        )

    def _render_boxes(self, frame: TrackingFrame, frame_image: np.ndarray):

        for td in frame.tracking_data:
            if not td.in_zone:
                # Render only objects that are not in the counting zone
                continue

            self._render_single_box(frame_image, td, (0, 0, 255))

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

        self._inverse_text(
            frame_image,
            f"{self.mouse_pos[0]},{self.mouse_pos[1]}",
            (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
            MOUSE_COLOR,
            1,
            0.5,
        )

    def _render_frame(self, frame: TrackingFrame, frame_image: np.ndarray):
        if self.show_zones:
            self._render_zones(frame, frame_image)
        if self.show_light:
            self._render_traffic_light(frame, frame_image)
        if self.show_text:
            self._render_text(frame, frame_image)
        if self.show_boxes:
            self._render_boxes(frame, frame_image)
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
