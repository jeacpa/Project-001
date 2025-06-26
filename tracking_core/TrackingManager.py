import bisect
import time
from typing import List, Optional

from cv2 import VideoCapture
import cv2
import numpy as np
from ultralytics import YOLO
from tracking_core.EventManager import EventManager
from tracking_core.RingBuffer import RingBuffer
from tracking_core.SimpleCounter import SimpleCounter, Tuple
from constants import (
    LIGHT_PHASES_BY_TIME_OFFSET,
    LIGHT_PHASES_TIMES,
    ZONE_CLEAR_CAR_COUNT,
    ZONE_CLEAR_COUNTDOWN_SEC,
)
from tracking_core.structures import (
    LightColor,
    TrackingData,
    TrackingFrame,
)


class TrackingManager:
    _cap: VideoCapture
    _writer: Optional[VideoCapture]
    _video_width: float
    _video_height: float
    _video_fps: int
    _last_frame_written: int
    _current_frame_index: int
    _realtime_frame_index: int
    _current_frame: Optional[TrackingFrame]
    _frame_skipping: bool
    _use_frame: bool
    _is_live: bool
    _tracking_classes: List[int]
    _model: YOLO
    _yolo_model_name: str
    _count_zone: List[Tuple[int, int]]
    _event_manager: EventManager
    _start_time: float
    _light_color: LightColor
    _light_duration: float
    _last_green_duration: float
    _last_red_duration: float
    _light_change_time: float
    _frame_buffer: RingBuffer
    _no_delay: bool

    # Zone clear time is used for auto light change and can have 3 states:
    #
    # 0 - Not counting down because too many cars are still in the zone (i.e. > ZONE_CLEAR_CAR_COUNT)
    # > 0 - Counting down until we request a light change (car count must remain below ZONE_CLEAR_CAR_COUNT)
    # -1 - An auto light change has been requested and we are waiting for the next red light
    zone_clear_time: float

    def __init__(
        self,
        cap: VideoCapture,
        yolo_model_name: str,
        tracking_classes: List[int],
        count_zone: List[Tuple[int, int]],
        event_manager: EventManager,
        buffer_file_name: Optional[str] = None,
        output_path: Optional[str] = None,
        frame_skipping: bool = False,
        is_live: bool = False,
        no_delay=False,
    ):

        self._cap = cap
        self._tracking_classes = tracking_classes
        self._video_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._video_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._last_frame_written = 0
        self._current_frame_index = 0
        self._realtime_frame_index = 0
        self._current_frame = None
        self._frame_skipping = frame_skipping
        self._use_frame = False
        self._is_live = is_live
        self._yolo_model_name = yolo_model_name
        self._count_zone = count_zone
        self._event_manager = event_manager
        self._last_green_duration = 0.0
        self._last_red_duration = 0.0
        self._zone_clear_time = 0
        self._start_time = 0.0
        self._light_color = LightColor.RED
        self._light_duration = 0.0
        self._light_change_time = 0.0
        self._no_delay = no_delay

        if output_path:
            self._writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                self._video_fps,
                (int(self._video_width), int(self._video_height)),
            )
        else:
            self._writer = None

        if buffer_file_name:
            self._frame_buffer = RingBuffer(buffer_file_name, 1000)
        else:
            self._frame_buffer = None

        self._reset_model()

    @property
    def current_frame(self) -> Optional[TrackingFrame]:
        return self._current_frame

    def _reset_model(self):

        self._model = YOLO(self._yolo_model_name)
        self._model.overrides["verbose"] = (
            False  # Hack: Verbose false on model does not seem to silence output
        )

        self._counter = SimpleCounter(self._count_zone)

    def _extract_tracking_data(self, frame_image: np.ndarray) -> List[TrackingData]:

        results = self._model.track(
            frame_image, persist=True, classes=self._tracking_classes
        )

        res = results[0]

        boxes = res.boxes.xyxy.cpu().numpy()  # Bounding boxes
        id = res.boxes.id
        if id is not None:
            ids = id.cpu().numpy().astype(int)  # Track IDs
        else:
            ids = [-1] * len(boxes)

        classes = res.boxes.cls.cpu().numpy().astype(int)  # Class indices

        return [
            TrackingData(box=list(map(int, box)), id=ids[idx], class_id=classes[idx])
            for idx, box in enumerate(boxes)
        ]

    # Returns phase and duration of phase
    def _get_light_phase(self, time_offset: float):

        position = bisect.bisect_right(LIGHT_PHASES_TIMES, time_offset)
        if position == 0:
            # If before first just use first
            return LIGHT_PHASES_BY_TIME_OFFSET[0][1], 0.0
        elif position >= len(LIGHT_PHASES_BY_TIME_OFFSET):
            # if after last then just use last
            return LIGHT_PHASES_BY_TIME_OFFSET[-1][1], 0.0
        else:
            # Otherwise using the previous
            return (
                LIGHT_PHASES_BY_TIME_OFFSET[position - 1][1],
                LIGHT_PHASES_BY_TIME_OFFSET[position][0]
                - LIGHT_PHASES_BY_TIME_OFFSET[position - 1][0],
            )

    def _analyze_light_change(self, time_offset: float):

        new_light_color, light_duration = self._get_light_phase(time_offset)

        if new_light_color == self._light_color:
            return

        self._light_change_time = time_offset
        if new_light_color == LightColor.GREEN:
            count_at_light = self._counter.in_count()

            self._event_manager.write_event(
                "red_to_green",
                {"seconds": self._last_red_duration, "in_zone": count_at_light},
            )
            self._last_green_duration = light_duration

        elif new_light_color == LightColor.RED:
            passed_through_light = self._counter.passed_through_count()
            count_at_light = self._counter.in_count()

            self._event_manager.write_event(
                "green_to_red",
                {
                    "seconds": self._last_green_duration,
                    "in_zone": count_at_light,
                    "exited_zone": passed_through_light,
                },
            )

            # Reset model to reset counting
            self._reset_model()
            self._zone_clear_time = 0
            self._last_red_duration = light_duration

        self._light_color = new_light_color
        self._light_duration = light_duration

    def _analyze_auto_light_change(self, time_offset: float):
        # We only currently check for turning green to red so a green light
        # or we can skip if we've already requested a change
        if self._light_color != LightColor.GREEN or self._zone_clear_time < 0:
            return

        # If car count above threshold then reset zone clear
        if self._counter.in_count() >= ZONE_CLEAR_CAR_COUNT:
            self._zone_clear_time = 0
            return

        # Set zone clear time so we start counting down
        if self._zone_clear_time == 0:
            # Start countdown
            self._zone_clear_time = time_offset
        else:
            elapsed = (time_offset - self._zone_clear_time) / 1000
            if elapsed > ZONE_CLEAR_COUNTDOWN_SEC:
                self._zone_clear_time = -1
                self._event_manager.write_event(
                    "auto_green_to_red",
                    {},
                )

    def _analyze_frame_image(
        self, frame_image: np.ndarray, time_offset: float
    ) -> List[TrackingData]:

        tracking_data = self._extract_tracking_data(frame_image)
        self._counter.process(tracking_data)

        self._analyze_light_change(time_offset)
        self._analyze_auto_light_change(time_offset)

        return tracking_data

    def _load_frame_from_stream(self):

        if not self._cap.isOpened():
            return

        # If we are frame skipping the burn a frame
        if self._frame_skipping:
            self._use_frame = not self._use_frame
            if not self._use_frame:
                self._cap.grab()

        start = time.time()

        success, frame_image = self._cap.read()
        if not success:
            self._current_frame = None
            return
            # raise VideoReadException("Could not read from video")

        self._realtime_frame_index += 1

        # If we are live then use current time
        if self._is_live:
            time_offset = (time.time() - self._start_time) * 1000.0
        else:
            time_offset = self._cap.get(cv2.CAP_PROP_POS_MSEC)

        tracking_data = self._analyze_frame_image(frame_image, time_offset)

        end = time.time()

        frame = TrackingFrame(
            time_offset=time_offset,
            frame_index=self._realtime_frame_index,
            tracking_data=tracking_data,
            raw_frame=frame_image,
            zone_clear_time=self._zone_clear_time,
            in_count=self._counter.in_count(),
            passed_through_count=self._counter.passed_through_count(),
            light_color=self._light_color,
            light_duration=self._light_duration,
            light_change_time=self._light_change_time,
            frame_processing_time_ms=round((end - start) * 1000.0),
        )

        if self._frame_buffer:
            self._frame_buffer.store(self._realtime_frame_index, frame)

        self._current_frame = frame

    def _load_frame_from_buffer(self):
        if self._frame_buffer and self._frame_buffer.item_exists(
            self._current_frame_index
        ):
            self._current_frame = self._frame_buffer.retrieve(self._current_frame_index)

    def _load_current_frame(self):

        start = time.time()

        # If we are ahead of realtime then grab from the feed
        if self._current_frame_index > self._realtime_frame_index:
            self._load_frame_from_stream()
        else:
            self._load_frame_from_buffer()

        end = time.time()

        # Artificially wait if needed to preserve video fps
        if not self._no_delay and not self._frame_skipping:
            wanted_frame_time = 1 / self._video_fps
            actual_frame_time = end - start

            if actual_frame_time < wanted_frame_time:
                time.sleep(wanted_frame_time - actual_frame_time)

    # Advances frame if not paused, loads frame data, and waits if needed to maintain fps
    # Note that it may be confusing to call advance_frame(True) as the frame doesn't actually
    # advance but we do this to make sure the paused frame is loaded and for any fps delay
    def advance_frame(self, paused: bool):
        if self._current_frame_index == 0:
            self._start_time = time.time()

        if not paused:
            self._current_frame_index += 1
            self._current_frame = None

        self._load_current_frame()

    def go_back(self, sec: int):
        if not self._frame_buffer:
            return

        new_frame_index = self._current_frame_index - round(sec * self._video_fps)
        if new_frame_index > 0 and self._frame_buffer.item_exists(new_frame_index):
            self._current_frame_index = new_frame_index

    def go_forward(self, sec: int):
        if not self._frame_buffer:
            return

        new_frame_index = self._current_frame_index + round(sec * self._video_fps)
        self._current_frame_index = min(new_frame_index, self._realtime_frame_index + 1)

    def write_frame_to_video(self, frame: TrackingFrame, frame_image: np.ndarray):
        if self._writer is None or frame.frame_index <= self._last_frame_written:
            return

        self._writer.write(frame_image)
        self._last_frame_written = frame.frame_index

    def set_frame_skipping(self, frame_skipping: bool):
        self._frame_skipping = frame_skipping

    def close(self):
        if self._writer:
            self._writer.release()
