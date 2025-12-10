import asyncio
import signal
import threading
import time
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from webapp.backend.messaging import ControlRequest, MessageingClient
from webapp.backend.tracking_loop.memory_streamer import MemoryStreamer

stop_event = threading.Event()

async def lifespan(app: FastAPI):
    global stop_event

    yield

    stop_event.set()

app = FastAPI(lifespan=lifespan)

# Enable CORS so your frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

async def jpeg_stream(request: Request) -> AsyncGenerator[bytes, None]:

    # Create a separate consumer instance for this stream
    # This ensures each client has independent _last_seen_seq tracking
    consumer = MemoryStreamer()

    frame_count = 0
    last_status_ts = 0
    sleeps = 0
    start_time = time.time()
    consecutive_none_count = 0

    try:
        while not stop_event.is_set():
            if await request.is_disconnected():
                print("Client disconnected")
                break
    
            now = time.time()
            if (now - last_status_ts) > 2:
                # Periodic housekeeping (silent)
                elapsed = now - start_time
                sleeps = 0
                last_status_ts = now

            frame_bytes: Optional[bytes] = consumer.consume_frame()

            if frame_bytes is None:
                if not consumer.is_connected:
                    print("Producer disconnected, retry later")
                    break
                
                # Exponential backoff: if we keep getting None, sleep longer
                consecutive_none_count += 1
                sleep_time = min(0.01 * (1 + consecutive_none_count // 10), 0.1)  # Cap at 100ms
                sleeps += 1
                await asyncio.sleep(sleep_time)
                continue

            # Successfully got a frame
            frame_count += 1
            consecutive_none_count = 0  # Reset backoff counter

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
        consumer.close()
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

            client = MessageingClient()
            res = client.send_message(req)
            await ws.send_json(jsonable_encoder(res, by_alias=True))

    except WebSocketDisconnect:
        print("WS Client disconnected")

original_sigterm = signal.getsignal(signal.SIGTERM)

def handle_sigterm(signum, frame):
    stop_event.set()

    if callable(original_sigterm):
        original_sigterm(signum, frame)


def handle_sigint(signum, frame):
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigterm)
