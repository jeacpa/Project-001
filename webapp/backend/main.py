import os
import sys
from typing import Optional
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from tracking_core.structures import TrackingFrame

from tracking_core.EventManager import NullEventManager
from tracking_core.TrackingManager import TrackingManager
from constants import COUNT_ZONE, MODEL_NAME, TRACKING_CLASSES

app = FastAPI()

# Enable CORS so your frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/data")
def get_data():
    return {"message": "Hello from FastAPI"}


def gen_frames():
    cap = cv2.VideoCapture("../../Cashmere.MP4")

    tracking = TrackingManager(
        cap=cap,
        yolo_model_name=MODEL_NAME,
        tracking_classes=TRACKING_CLASSES,
        count_zone=COUNT_ZONE,
        event_manager=NullEventManager(),
        # output_path=OUTPUT_VIDEO,  # No output video for now
        # frame_skipping=self.half_frames,
        is_live=False,  # Not live, we are reading a video file
        # buffer_file_name=FRAME_BUFFER_FILE,
    )

    while True:
        frame: Optional[TrackingFrame]

        tracking.advance_frame(False)

        frame = tracking.current_frame.raw_frame

        if frame is None:
            break

        # frame_out = np.copy(frame.raw_frame)

        # self._render_frame(frame, frame_out)

        # self._show_frame(frame_out)

        # tracking.write_frame_to_video(frame, frame_out)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    tracking.close()
    cap.release()
    print("===============!")
    pass


@app.get("/")
def read_root():
    gen_frames()
    return {"message": "Hello FastAPI"}


@app.get("/video")
def stream_video():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )
