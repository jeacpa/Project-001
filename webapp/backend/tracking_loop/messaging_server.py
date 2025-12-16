from multiprocessing.connection import Listener
import socket
import threading


from constants import (
    IPC_ADDRESS,
)

from webapp.backend.messaging import ControlRequest, ControlResponse
import webapp.backend.globals as wb_globals
from webapp.backend.tracking_loop.tracking_loop import save_zone
from constants import TIME_INTERESTING_START_MS, TIME_GOOD_START_MS

messaging_server_thread: threading.Thread

def handle_control_request(req: ControlRequest) -> ControlResponse:
    action = req.action

    if action == "toggle_zones":
        wb_globals.control_state.show_zones = not wb_globals.control_state.show_zones
    elif action == "toggle_light":
        wb_globals.control_state.show_light = not wb_globals.control_state.show_light
    elif action == "toggle_text":
        wb_globals.control_state.show_text = not wb_globals.control_state.show_text
    elif action == "toggle_boxes":
        wb_globals.control_state.show_boxes = not wb_globals.control_state.show_boxes
    elif action == "toggle_pause":
        wb_globals.control_state.paused = not wb_globals.control_state.paused
    elif action == "set_paused":
        wb_globals.control_state.paused = True
    elif action == "fast_forward" and wb_globals.tracking:
        wb_globals.tracking.go_forward(1)
    elif action == "rewind" and wb_globals.tracking:
        wb_globals.tracking.go_back(1)
    elif action == "restart":
        wb_globals.reset_loop_requested = TIME_GOOD_START_MS
    elif action == "infojump":
        wb_globals.reset_loop_requested = TIME_INTERESTING_START_MS
    elif action == "cursor_pos":
        if req.x is not None and req.y is not None and wb_globals.tracking:
            wb_globals.tracking.set_cursor_pos((req.x, req.y))
    elif action == "select_box":
        if wb_globals.tracking:
            wb_globals.tracking.select_box_under_cursor()
    elif action == "set_zone":
        save_zone(req.count_zone)
        # Need to reset the loop so that
        # a) New zone is loaded
        # b) Counters are reset for cars in/out
        # c) Frame buffer is cleared because zone has changed
        wb_globals.reset_loop_requested = TIME_GOOD_START_MS

    return ControlResponse(state=wb_globals.control_state)

def messaging_client_handler(conn):
    try:
        while not wb_globals.stop_event.is_set():
            # poll for up to 0.5 seconds
            if conn.poll(0.5):
                try:
                    cmd: ControlRequest = conn.recv()
                except EOFError:
                    break

                conn.send(handle_control_request(cmd))
            # else: timed out, loop again and check stop_event
    finally:
        conn.close()


def messaging_server():
    try:
        listener = Listener(IPC_ADDRESS)
    except OSError as e:
        print(f"!!!! Messaging server failed to start on {IPC_ADDRESS}: {e}")
        return

    # Hack: set timeout on accept so we can check for stop_event
    listener._listener._socket.settimeout(1.0)

    print("Messaging server started on ", IPC_ADDRESS)

    try:
        while not wb_globals.stop_event.is_set():
            try:
                conn = listener.accept()  # now times out after 1s
            except socket.timeout:
                # just loop back and check stop_event
                continue
            except (OSError, EOFError):
                # likely shutting down
                break

            threading.Thread(
                target=messaging_client_handler,
                args=(conn,),
                daemon=True,
            ).start()
    finally:
        listener.close()    
    
    print("Messaging server stopped")

def start_messaging_server():
    global messaging_server_thread

    messaging_server_thread = threading.Thread(
        target=messaging_server,
        daemon=True,
    )
    messaging_server_thread.start()

def stop_messaging_server():
    global messaging_server_thread

    messaging_server_thread.join()
    messaging_server_thread = None

