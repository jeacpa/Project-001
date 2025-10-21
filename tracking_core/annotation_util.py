from typing import List, Optional, Tuple
import cv2
import numpy as np
from constants import (
    BLACK_LIGHT,
    BOX_LINE_THICKNESS,
    BOX_TEXT_COLOR,
    BOX_TEXT_SCALE,
    BOX_TEXT_THICKNESS,
    GREEN_LIGHT,
    INFO_TEXT_POS,
    LINE_COLOR,
    LINE_THICKNESS,
    RED_LIGHT,
    TEXT_COLOR,
    TEXT_LINE_HEIGHT,
    TEXT_SCALE,
    TEXT_THICKNESS,
    TRACKING_LABELS,
    YELLOW_LIGHT,
    ZONE_CLEAR_CAR_COUNT,
    ZONE_CLEAR_COUNTDOWN_SEC,
)
from tracking_core.draw_util import center_text, interpolate_color, inverse_text
from tracking_core.SimpleCounter import TrackingData
from tracking_core.structures import LightColor, TrackingFrame


def render_zones(
    frame: TrackingFrame, zone: List[Tuple[int, int]], frame_image: np.ndarray
):

    line_color = LINE_COLOR

    if frame.zone_clear_time > 0:
        elapsed = (frame.time_offset - frame.zone_clear_time) / 1000
        t = min(elapsed / ZONE_CLEAR_COUNTDOWN_SEC, 1.0)
        line_color = interpolate_color(LINE_COLOR, (0, 0, 255), t)
    elif frame.zone_clear_time == -1:
        # Flash every 500ms
        if frame.time_offset % 1000 < 500:
            line_color = (0, 0, 255)
        else:
            return

    cv2.polylines(
        frame_image,
        [np.array(zone, dtype=np.int32)],
        isClosed=True,
        color=line_color,
        thickness=LINE_THICKNESS,
    )


def render_text(
    frame: TrackingFrame, selected_id: Optional[int], frame_image: np.ndarray
):
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

    if selected_id is not None:
        out(f"Tracking ID: {selected_id}")


def render_info_text(frame: TrackingFrame, frame_image: np.ndarray):

    def out(text):
        inverse_text(
            frame_image,
            text,
            INFO_TEXT_POS,
            (50, 50, 50),
            TEXT_THICKNESS,
            TEXT_SCALE,
            (255, 255, 255),
        )

    if frame.zone_clear_time > 0:
        elapsed = (frame.time_offset - frame.zone_clear_time) / 1000

        out(f"Less than {ZONE_CLEAR_CAR_COUNT} cars for {round(elapsed)} seconds")
    elif frame.zone_clear_time == -1:
        out(
            f"Less than {ZONE_CLEAR_CAR_COUNT} cars for {ZONE_CLEAR_COUNTDOWN_SEC} seconds, light should have changed!"
        )


def render_traffic_light(frame: TrackingFrame, frame_image: np.ndarray):
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
            frame.light_color == LightColor.RED or frame.light_color == LightColor.GREEN
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

        center_text(
            frame_image,
            red_text,
            (x + w // 2, y + h // 4),
            color=green_color,
            scale=0.6,
        )
        center_text(
            frame_image,
            green_text,
            (x + w // 2, y + 3 * h // 4),
            color=red_color,
            scale=0.6,
        )


def render_single_box(frame_image: np.ndarray, td: TrackingData, color):
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

    inverse_text(
        frame_image,
        label,
        (x1, y1),
        BOX_TEXT_COLOR,
        BOX_TEXT_THICKNESS,
        BOX_TEXT_SCALE,
    )


def render_selected_box(frame_image: np.ndarray, td: TrackingData):
    x1, y1, x2, y2 = td.box

    cv2.polylines(
        frame_image,
        [
            np.array(
                [
                    (x1 - 5, y1 - 5),
                    (x2 + 5, y1 - 5),
                    (x2 + 5, y2 + 5),
                    (x1 - 5, y2 + 5),
                ],
                dtype=np.int32,
            )
        ],
        isClosed=True,
        color=(255, 50, 50),
        thickness=BOX_LINE_THICKNESS,
    )


def render_boxes(
    frame: TrackingFrame, selected_id: Optional[int], frame_image: np.ndarray
):

    for td in frame.tracking_data:
        if td.id == selected_id:
            render_selected_box(frame_image, td)

        if td.under_cursor:
            # Always show box under cursor in white
            render_single_box(frame_image, td, (255, 255, 255))
            continue

        if not td.in_zone:
            # Render only objects that are not in the counting zone
            continue

        render_single_box(
            frame_image, td, (255, 0, 0) if td.under_cursor else (0, 0, 255)
        )


def render_cursor(frame_image: np.ndarray, cursor_pos: Tuple[int, int]):
    if cursor_pos is None:
        return

    cv2.circle(frame_image, cursor_pos, 10, (0, 255, 0), -1)
