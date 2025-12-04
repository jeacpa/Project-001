import asyncio
import signal
import threading
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from webapp.backend.messaging import ControlRequest, MessageingClient
from webapp.backend.tracking_loop.memory_streamer import MemoryStreamer

stop_event = threading.Event()

mem_stream: Optional[MemoryStreamer] = None

async def lifespan(app: FastAPI):
    global mem_stream, stop_event

    mem_stream = MemoryStreamer()

    yield

    stop_event.set()

    mem_stream.close()

app = FastAPI(lifespan=lifespan)

# Enable CORS so your frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

async def jpeg_stream(request: Request) -> AsyncGenerator[bytes, None]:

    frame_count = 0

    try:
        while not stop_event.is_set():
            if await request.is_disconnected():
                print("Client disconnected")
                break
    
            frame_bytes: Optional[bytes] = mem_stream.consume_frame()

            if frame_bytes is None:
                # See if stream is still connected
                if not mem_stream.is_connected:
                    print("Producer disconnected, retry later")
                    break

                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            
            # print(f"!!! Time to encode frame #{frame_count}: ", (end - start) * 1000, "ms")

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
