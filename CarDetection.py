# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")
# model.to('cuda')

# results = model("Cashmere.MP4", save=True, show=True)

import cv2
import numpy as np
import time
from ultralytics import YOLO

# ------------------------
# Load the Ultralytics YOLO model
# ------------------------
model = YOLO("yolo11n.pt")  # You can change to a different YOLOv8 model if needed

# Define the class names to consider (based on COCO dataset)
# For vehicles: car, bus, truck, motorcycle; and for pedestrians: person
vehicle_classes = {"car", "bus", "truck", "motorcycle"}
pedestrian_class = "person"

# ------------------------
# Initialize video capture
# ------------------------
video_path = "Cashmere.MP4"  # Replace with your video file path
print(video_path)

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(framecount)

# Define a counting line (here, a horizontal line at 50% of frame height)
line_y = int(frame_height * 0.3)
line_color = (0, 255, 255)  # Yellow line
line_thickness = 2

# ------------------------
# Initialize trackers, counters, and timer
# ------------------------
# Using a simple centroid-based tracker:
# Format: {object_id: (centroid, counted_flag)}
trackers = {}
next_object_id = 0
line_count = 0

# Countdown timer settings
timer_duration = 10  # seconds
last_reset_time = time.time()  # resets each time a vehicle crosses the line


def get_centroid(box):
    """Compute centroid given a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def update_trackers(detections, trackers, next_object_id, distance_threshold=100):
    """
    A simple tracker that associates current detections with existing ones
    based on Euclidean distance. If no match is found, a new tracker is created.
    """
    updated_trackers = {}
    for box, label in detections:
        centroid = get_centroid(box)
        best_id = None
        min_dist = float("inf")
        for obj_id, (prev_centroid, counted) in trackers.items():
            dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if dist < distance_threshold and dist < min_dist:
                best_id = obj_id
                min_dist = dist
        if best_id is None:
            best_id = next_object_id
            next_object_id += 1
        # Preserve the counted flag if it exists
        counted_flag = trackers.get(best_id, (None, False))[1]
        updated_trackers[best_id] = (centroid, counted_flag)
        print(best_id)
    return updated_trackers, next_object_id


# ------------------------
# Main loop for video processing
# ------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the YOLO model on the current frame
    results = model(frame)[0]  # Get results for the current frame

    # Prepare a list to store detections (only vehicles for tracking/counting)
    detections_vehicle = []

    # Ensure there are detections
    if results.boxes is not None and len(results.boxes) > 0:
        # Extract bounding boxes, confidences, and class indices
        boxes = results.boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        confidences = results.boxes.conf.cpu().numpy()
        class_indices = results.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            label = model.names[int(class_indices[i])]
            conf = confidences[i]
            if conf < 0.5:
                continue  # Filter out low-confidence detections
            x1, y1, x2, y2 = box.astype(int)
            # Check if the detected object is a vehicle or pedestrian
            if label in vehicle_classes or label == pedestrian_class:
                # Set color based on type
                color = (0, 255, 0) if label in vehicle_classes else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                # Draw the centroid for visualization
                centroid = get_centroid((x1, y1, x2, y2))
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                # Only track vehicles (not pedestrians) for counting
                if label in vehicle_classes:
                    detections_vehicle.append(((x1, y1, x2, y2), label))

    # Update the tracker with the current vehicle detections
    trackers, next_object_id = update_trackers(
        detections_vehicle, trackers, next_object_id
    )

    # Check for vehicles crossing the counting line (assumes vehicles are moving downward)
    for obj_id, (centroid, counted) in trackers.items():
        cx, cy = centroid
        if not counted and cy > line_y:
            line_count += 1
            # Reset the countdown timer when a vehicle is counted
            last_reset_time = time.time()
            trackers[obj_id] = (centroid, True)

    # Calculate the countdown timer value
    elapsed = time.time() - last_reset_time
    countdown = max(0, timer_duration - elapsed)

    # Display the timer and vehicle count on the frame
    cv2.putText(
        frame,
        f"Timer: {int(countdown)}s",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Vehicles Passed: {line_count}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.line(frame, (0, line_y), (frame_width, line_y), line_color, line_thickness)

    cv2.imshow("Intersection Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
