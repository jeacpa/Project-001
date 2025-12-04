
import signal
from webapp.backend.globals import control_state, stop_event
from webapp.backend.tracking_loop.messaging_server import start_messaging_server, stop_messaging_server
from webapp.backend.tracking_loop.tracking_loop import load_zone, run_loop

original_sigterm = signal.getsignal(signal.SIGTERM)


def handle_sigterm(signum, frame):
    stop_event.set()

    if callable(original_sigterm):
        original_sigterm(signum, frame)


def handle_sigint(signum, frame):
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigterm)

stop_event.clear()

start_messaging_server()

# Insure zone is loaded
control_state.count_zone = load_zone()

run_loop()

stop_messaging_server()
