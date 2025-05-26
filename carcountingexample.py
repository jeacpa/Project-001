import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file or camera stream
cap = cv2.VideoCapture("Cashmere.MP4")  # Replace with your video path or 0 for webcam

# Initialize variables
car_count = 0
tracked_cars = {}  # Dictionary to store car positions and IDs
counted_cars = set()  # Set to track cars that have already been counted
line_position = 600  # Y-coordinate of the imaginary line


def has_crossed_line(prev_pos, curr_pos, line_y):
    """Check if car crossed the line from top to bottom"""
    if prev_pos is None or curr_pos is None:
        return False
    prev_y = prev_pos[1]
    curr_y = curr_pos[1]
    return prev_y < line_y and curr_y >= line_y


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Get current frame detections
    current_detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 2:  # Class 2 is car
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                current_detections.append((center_x, center_y))

    # Update tracked cars and count crossings
    new_tracked_cars = {}
    for i, (curr_x, curr_y) in enumerate(current_detections):
        car_id = None
        min_dist = float("inf")

        # Match with previous detections
        for prev_id, prev_pos in tracked_cars.items():
            dist = np.sqrt((curr_x - prev_pos[0]) ** 2 + (curr_y - prev_pos[1]) ** 2)
            if dist < min_dist and dist < 100:  # Max distance threshold
                min_dist = dist
                car_id = prev_id

        # If no match found, assign new ID
        if car_id is None:
            car_id = len(tracked_cars) + len(counted_cars)

        # Check if car crossed the line and hasn't been counted yet
        prev_pos = tracked_cars.get(car_id)
        if (
            has_crossed_line(prev_pos, (curr_x, curr_y), line_position)
            and car_id not in counted_cars
        ):
            car_count += 1
            counted_cars.add(car_id)

        new_tracked_cars[car_id] = (curr_x, curr_y)

    # Update tracked cars
    tracked_cars = new_tracked_cars

    # Draw the counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)

    # Draw detections and IDs with count status
    for car_id, (x, y) in tracked_cars.items():
        color = (0, 255, 0) if car_id in counted_cars else (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(
            frame, f"ID: {car_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # Display counter
    cv2.putText(
        frame,
        f"Cars counted: {car_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    # Show the frame
    cv2.imshow("Car Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Total cars counted: {car_count}")
