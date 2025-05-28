import asyncio
import bisect
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import cv2

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import SolutionAnnotator
from ultralytics import solutions


import logging

from Clients.sql import SqlClient, SqlTransaction
import struct
from SimpleCounter import SimpleCounter
from structures import TrackingData, TrackingEvent
from util import utc_now

logging.getLogger("ultralytics").setLevel(logging.ERROR)


class VideoReadException(Exception):
    pass


class LightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3


class IntersectionDirection(Enum):
    N = "N"
    S = "S"
    E = "E"
    W = "W"
    NW = "NW"
    NE = "NE"
    SW = "SW"
    SE = "SE"


WINDOW_NAME = "YOLO11 Tracking"

MODEL_NAME = "yolo11n.pt"

OUTPUT_VIDEO = "vidout.mp4"

INTERSECTION_ID = "001"
INTERSECTION_DIRECTION = IntersectionDirection.E

COUNT_ZONE = [(506, 360), (910, 316), (1521, 662), (778, 746)]

LANE_COUNT_ZONES = {
    "1": [(506, 360), (656, 341), (1025, 721), (778, 744)],
    "2": [(606, 351), (710, 337), (1141, 703), (973, 723)],
    "3": [(669, 339), (754, 330), (1250, 692), (1100, 709)],
    "4": [(733, 334), (806, 325), (1262, 680), (1199, 698)],
    "5": [(790, 327), (910, 313), (1515, 657), (1333, 683)],
}
YELLOW_LIGHT = (0, 255, 255)
GREEN_LIGHT = (0, 255, 0)
BLACK_LIGHT = (65, 74, 76)
RED_LIGHT = (0, 0, 255)

TEXT_COLOR = (255, 255, 0)
TEXT_LINE_HEIGHT = 40
TEXT_SCALE = 1
TEXT_THICKNESS = 2

# Bicycle, Car, Motorcycle, Bus, Truck
TRACKING_CLASSES = [1, 2, 3, 5, 7]
TRACKING_LABELS = {1: "Bike", 2: "Car", 3: "Moto", 5: "Bus", 7: "Truck"}
LINE_COLOR = (0, 255, 255)  # Yellow line
LINE_THICKNESS = 3

BOX_LINE_COLOR = (0, 165, 255)
BOX_TEXT_COLOR = (0, 165, 255)
BOX_LINE_THICKNESS = 2
BOX_TEXT_THICKNESS = 1
BOX_TEXT_SCALE = 0.4

MOUSE_COLOR = (0, 0, 255)

LIGHT_PHASES_BY_FRAME = [
    (0, LightColor.RED),
    (900, LightColor.GREEN),
    (3200, LightColor.YELLOW),
    (3300, LightColor.RED),
    (6180, LightColor.GREEN),
]
LIGHT_PHASES_FRAMES = [frame_color[0] for frame_color in LIGHT_PHASES_BY_FRAME]


# Hack: We want to draw ourselves, thank you very much
def stub_function(*args, **kwargs):
    pass


SolutionAnnotator.draw_region = stub_function
SolutionAnnotator.box_label = stub_function
SolutionAnnotator.display_analytics = stub_function


class Expiriment:
    mode: YOLO
    # counter: solutions.ObjectCounter
    counter: SimpleCounter
    video_path: str
    cap: cv2.VideoCapture
    show_text: bool
    show_boxes: bool
    show_zones: bool
    show_light: bool
    show_mouse: bool
    vboxsize: int
    should_exit: bool
    window_exists: bool
    light_color: LightColor
    light_duration: float
    tracking_results: Results
    vehicle_count: int
    tracking_data: Dict[int, TrackingData]
    time_offset: float
    light_change_time: float
    frame_offset: int
    mouse_pos: Tuple[int, int]
    saved_counted: set
    save_events: bool
    start_offset_ms: int
    video_width: float
    video_height: float
    video_fps: int

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
    ):
        self.video_path = video_path
        self.vboxsize = 1
        self.show_boxes = show_boxes
        self.show_zones = show_zones
        self.show_light = show_light
        self.show_text = show_text
        self.show_mouse = show_mouse
        self.should_exit = False
        self.window_exists = False
        self.light_color = LightColor.RED
        self.saved_counted = set()
        self.save_events = save_events
        self.start_offset_ms = start_offset_ms
        self.light_change_time = 0
        self.light_duration = 0

        self._init_model()

    def _init_model(self):

        self.model = YOLO(MODEL_NAME)
        self.model.overrides["verbose"] = (
            False  # Hack: Verbose false on model does not seem to silence output
        )

        self.counter = SimpleCounter(COUNT_ZONE)

    def _render_zones(self, frame):
        cv2.polylines(
            frame,
            [np.array(COUNT_ZONE, dtype=np.int32)],
            isClosed=True,
            color=LINE_COLOR,
            thickness=LINE_THICKNESS,
        )

    def _render_text(self, frame):
        current_y = 50

        def out(text):
            nonlocal current_y
            cv2.putText(
                frame,
                text,
                (10, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
            )
            current_y += TEXT_LINE_HEIGHT

        out(f"Frame: {self.frame_offset}")
        out(f"Time: {(self.time_offset/1000):.2f}s")
        out(f"Objects in: {self.counter.in_count()}")
        out(f"Objects out: {self.counter.passed_through_count()}")

    def _render_help(self, frame):
        current_y = 500

        def out(text):
            nonlocal current_y
            cv2.putText(
                frame,
                text,
                (20, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
            current_y += 30

        cv2.rectangle(frame, (10, 480), (210, 630), TEXT_COLOR, cv2.FILLED)
        out("Show Stoplight (T)")
        out("Show Zone (L)")
        out("Show Mouse (M)")
        out("Show Boxes (B)")
        out("Quit (Q)")

    def _center_text(self, frame, text, xy_center, color, thickness=1, scale=1.0):
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        cv2.putText(
            frame,
            text,
            (xy_center[0] - text_width // 2, xy_center[1] + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _inverse_text(self, frame, text, xy, color, thickness=1, scale=1.0):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        baseline += 2
        box_coords1 = (xy[0] - 4, xy[1] - text_height - 4)
        box_coords2 = (xy[0] + text_width + 4, xy[1] + baseline)

        cv2.rectangle(frame, box_coords1, box_coords2, color, cv2.FILLED)
        cv2.putText(
            frame,
            text,
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _render_traffic_light(self, frame):
        x, y, w, h = 1800, 1, 80, 300  # x, y coordinates, width, height
        padding = 7
        circle_radius = (w - 2 * padding) // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

        cv2.circle(
            frame,
            (x + w // 2, y + h // 4),
            circle_radius,
            RED_LIGHT if self.light_color == LightColor.RED else BLACK_LIGHT,
            -1,
        )

        cv2.circle(
            frame,
            (x + w // 2, y + h // 2),
            circle_radius,
            YELLOW_LIGHT if self.light_color == LightColor.YELLOW else BLACK_LIGHT,
            -1,
        )
        cv2.circle(
            frame,
            (x + w // 2, y + 3 * h // 4),
            circle_radius,
            GREEN_LIGHT if self.light_color == LightColor.GREEN else BLACK_LIGHT,
            -1,
        )

        if (
            self.light_change_time
            and self.light_duration
            and (
                self.light_color == LightColor.RED
                or self.light_color == LightColor.GREEN
            )
        ):

            elapsed = round((self.time_offset - self.light_change_time) / 1000)
            remaining = max(0, round(self.light_duration / 1000) - elapsed)
            # remaining = max(0, round((self.light_duration - (self.time_offset - self.light_change_time))/1000))

            if self.light_color == LightColor.RED:
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
                frame, red_text, (x + w // 2, y + h // 4), color=green_color, scale=0.6
            )
            self._center_text(
                frame,
                green_text,
                (x + w // 2, y + 3 * h // 4),
                color=red_color,
                scale=0.6,
            )

    def _render_single_box(self, frame, td: TrackingData, color):
        x1, y1, x2, y2 = td.box
        label = TRACKING_LABELS[td.class_id]
        label = f"{label}/{td.id}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            BOX_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        self._inverse_text(
            frame,
            label,
            (x1, y1),
            BOX_TEXT_COLOR,
            BOX_TEXT_THICKNESS,
            BOX_TEXT_SCALE,
        )

    def _render_boxes(self, frame):

        for td in self.counter.in_objects():
            self._render_single_box(frame, td, (0, 0, 255))

    def _render_mouse(self, frame):
        if not hasattr(self, "mouse_pos"):
            return
        cv2.line(
            frame,
            (self.mouse_pos[0] - 20, self.mouse_pos[1]),
            (self.mouse_pos[0] + 20, self.mouse_pos[1]),
            MOUSE_COLOR,
            2,
        )
        cv2.line(
            frame,
            (self.mouse_pos[0], self.mouse_pos[1] - 20),
            (self.mouse_pos[0], self.mouse_pos[1] + 20),
            MOUSE_COLOR,
            2,
        )

        self._inverse_text(
            frame,
            f"{self.mouse_pos[0]},{self.mouse_pos[1]}",
            (self.mouse_pos[0] + 10, self.mouse_pos[1] - 10),
            MOUSE_COLOR,
            1,
            0.5,
        )

    # Returns phase and duration of phase
    def _get_light_phase(self):
        position = bisect.bisect_right(LIGHT_PHASES_FRAMES, self.frame_offset)
        if position == 0:
            # If before first just use first
            return LIGHT_PHASES_BY_FRAME[0][1], 0
        elif position >= len(LIGHT_PHASES_BY_FRAME):
            # if after last then just use last
            return LIGHT_PHASES_BY_FRAME[-1][1], 0
        else:
            # Otherwise using the previous
            return (
                LIGHT_PHASES_BY_FRAME[position - 1][1],
                (
                    (
                        LIGHT_PHASES_BY_FRAME[position][0]
                        - LIGHT_PHASES_BY_FRAME[position - 1][0]
                    )
                    / float(self.video_fps)
                )
                * 1000.0,
            )

    def _extract_tracking_data(self, frame):
        results = self.model.track(frame, persist=True, classes=TRACKING_CLASSES)

        res = results[0]

        boxes = res.boxes.xyxy.cpu().numpy()  # Bounding boxes
        ids = res.boxes.id.cpu().numpy().astype(int)  # Track IDs
        classes = res.boxes.cls.cpu().numpy().astype(int)  # Class indices

        self.tracking_data = [
            TrackingData(box=list(map(int, box)), id=ids[idx], class_id=classes[idx])
            for idx, box in enumerate(boxes)
        ]

    def _analyze_frame(self, frame):

        self._extract_tracking_data(frame)
        self.counter.process(self.tracking_data)

        self.time_offset = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.frame_offset = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        new_light_color, light_duration = self._get_light_phase()

        if new_light_color != self.light_color:
            self.light_change_time = self.time_offset
            if new_light_color == LightColor.GREEN:
                count_at_light = self.counter.in_count()
                print(f"Waiting for green: {count_at_light}")
            elif new_light_color == LightColor.RED:
                passed_through_light = self.counter.passed_through_count()
                count_at_light = self.counter.in_count()
                # Reset model to reset counting
                self._init_model()
                print(
                    f"Passed through light: {passed_through_light}, Count at light: {count_at_light}"
                )

        self.light_color = new_light_color
        self.light_duration = light_duration

        # Let cv render boxes
        if self.show_boxes:
            self._render_boxes(frame)
            # frame = results[0].plot(line_width=self.vboxsize, conf=True, boxes=boxes)
        if self.show_zones:
            self._render_zones(frame)
        if self.show_light:
            self._render_traffic_light(frame)
        if self.show_text:
            self._render_text(frame)
        if self.show_mouse:
            self._render_mouse(frame)

        self._render_help(frame)

        return frame

    def _check_input(self):

        key = cv2.waitKey(1)
        if key == ord("1"):
            self.vboxsize = 1
        elif key == ord("2"):
            self.vboxsize = 2
        elif key == ord("b"):
            self.show_boxes = not self.show_boxes
        elif key == ord("l"):
            self.show_zones = not self.show_zones
        elif key == ord("t"):
            self.show_light = not self.show_light
        elif key == ord("m"):
            self.show_mouse = not self.show_mouse
        elif key == ord("q"):
            self.should_exit = True

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            self.should_exit = True

    def _handle_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

    def _write_batch_events(self, conn: SqlClient, events: List[TrackingEvent]):

        conn.insert_batch(
            "tbl_event",
            ["created_at", "name", "attributes"],
            [(utc_now(), te.name, te.attributes) for te in events],
        )

    def _write_events(self, conn: SqlClient):
        return
        to_save: List[struct] = []
        for td in self.tracking_data:
            if td.counted and td.id not in self.saved_counted:
                to_save.append(td)

        conn.insert_batch(
            "tbl_event",
            ["created_at", "name", "attributes"],
            [(utc_now(), "counted", {"id": td.id}) for td in to_save],
        )

        for td in to_save:
            self.saved_counted.add(td.id)

    def _open_capture(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_offset_ms)

    def _open_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        return cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            self.video_fps,
            (int(self.video_width), int(self.video_height)),
        )

    def _run_video_analysis(self, conn: SqlClient = None):

        print("Loading video...")

        self._open_capture()

        writer = self._open_writer()

        with SqlClient() as conn:
            while self.cap.isOpened() and not self.should_exit:
                # Read a frame from the video
                success, frame = self.cap.read()

                if not success:
                    raise VideoReadException("Could not read from video")

                frame_out = self._analyze_frame(frame)

                if not self.window_exists:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, 1920, 1080)
                    cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)

                    self.window_exists = True

                cv2.imshow(WINDOW_NAME, frame_out)
                writer.write(frame_out)
                self._check_input()

                if conn:
                    self._write_events(conn)

        writer.release()
        self.cap.release()

        cv2.destroyAllWindows()

    def process(self):

        if self.save_events:
            with SqlClient() as conn:
                self._run_video_analysis(conn)
        else:
            self._run_video_analysis()


# Start 27s in to skip initial cross traffic
exp = Expiriment(
    video_path="Cashmere.MP4", start_offset_ms=27000, show_light=True, show_boxes=True
)

exp.process()
