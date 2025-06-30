import cv2
import numpy as np


def inverse_text(frame_image: np.ndarray, text, xy, color, thickness=1, scale=1.0):
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


def interpolate_color(start_color, end_color, t: float) -> tuple[int, int, int]:
    """Linearly interpolate between two BGR colors"""
    return tuple(
        [int(start + (end - start) * t) for start, end in zip(start_color, end_color)]
    )


def center_text(
    frame_image: np.ndarray, text, xy_center, color, thickness=1, scale=1.0
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
