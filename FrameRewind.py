import cv2

from ultralytics import YOLO


frame_counter = 0
# Define region points
# region_points = [(485, 250), (677, 205), (855, 755), (1550, 700)]
region_points = [(485, 250), (670, 205), (1550, 700), (840, 755)]

# Define a counting line (here, a horizontal line at 50% of frame height)

line_color = (0, 255, 255)  # Yellow line


model = YOLO("yolo11n.pt")  # use 'cuda' for GPU

# Open the video
cap = cv2.VideoCapture("Cashmere.MP4")
print("hello")
if not cap.isOpened():
    print("Error: Could not open video.")

car_counter = 0
tracked_cars = {}  # Dictionary to track cars across frames

while True:
    ret, frame = cap.read()
    # print(frame)
    print(frame.)
    if not ret:
        break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

