from pathlib import Path
from tracking_core.structures import IntersectionDirection, LightColor


LIGHT_PHASES_BY_TIME_OFFSET = [
    (0, LightColor.RED),
    (30000, LightColor.GREEN),
    (106000, LightColor.YELLOW),
    (110000, LightColor.RED),
    (206000, LightColor.GREEN),
    (276000, LightColor.YELLOW),
    (280000, LightColor.RED),
    (375000, LightColor.GREEN),
]
LIGHT_PHASES_TIMES = [time_color[0] for time_color in LIGHT_PHASES_BY_TIME_OFFSET]


WINDOW_NAME = "YOLO11 Tracking"

MODEL_NAME = "yolo11n.pt"

OUTPUT_VIDEO = "vidout.mp4"

FRAME_BUFFER_FILE = "buffer.dat"

INTERSECTION_ID = "001"
INTERSECTION_DIRECTION = IntersectionDirection.E

# Default area of incoming traffic to count
COUNT_ZONE = [(506, 360), (910, 316), (1521, 662), (778, 746)]

ZONE_FILE = Path("zone.json")

# Location to display information text
INFO_TEXT_POS = (778, 850)

# Number of cars or lower which will trigger a countdown until we request a turn of the light
ZONE_CLEAR_CAR_COUNT = 5

# Amount of time zone must contain ZONE_CLEAR_CAR_COUNT or less cars before we request a turn of the light
ZONE_CLEAR_COUNTDOWN_SEC = 8


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

IPC_ADDRESS = "127.0.0.1:9002"

TIME_GOOD_START_MS = 27000
TIME_INTERESTING_START_MS = 55000
