import cv2
import numpy as np

# ------------------------
# Load YOLO model files
# ------------------------
# Replace these file paths with the paths to your YOLOv3 weights, config, and class names files.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
# Get output layers names from YOLO
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define which classes to consider as vehicles (and pedestrians separately)
vehicle_classes = {"car", "motorbike", "bus", "truck"}
pedestrian_class = "person"

# ------------------------
# Initialize video capture
# ------------------------
video_path = "Cashmere.MP4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define a counting line (here, a horizontal line at 50% of frame height)
line_y = int(frame_height * 0.5)
line_color = (0, 255, 255)  # Yellow line
line_thickness = 2

# ------------------------
# Initialize trackers and counters
# ------------------------
# A simple dictionary to keep track of vehicle centroids and whether they’ve been counted:
# Format: {object_id: (centroid, counted_flag)}
trackers = {}
next_object_id = 0
line_count = 0


def get_centroid(box):
    """Calculate centroid given a bounding box [x, y, w, h]."""
    x, y, w, h = box
    return (int(x + w / 2), int(y + h / 2))


def update_trackers(detections, trackers, next_object_id, distance_threshold=50):
    """
    A simple tracker that associates current detections with previous centroids
    based on Euclidean distance. If no existing tracker is close enough, a new ID is assigned.
    """
    updated_trackers = {}
    # For each detection, try to match with an existing tracker.
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
        # If already tracked, preserve its counted flag
        counted_flag = trackers.get(best_id, (None, False))[1]
        updated_trackers[best_id] = (centroid, counted_flag)
    return updated_trackers, next_object_id


# ------------------------
# Main loop for video processing
# ------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    height, width, channels = frame.shape

    # Prepare input blob for YOLO and perform forward pass
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process each detection from YOLO
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Scale detection coordinates back to frame size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-max suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # List to store detections for tracking (only vehicles)
    vehicle_detections = []

    # Loop over the filtered detections
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = classes[class_ids[i]]
            box = boxes[i]
            # Process only vehicles and pedestrians
            if label in vehicle_classes or label == pedestrian_class:
                color = (0, 255, 0) if label in vehicle_classes else (255, 0, 0)
                cv2.rectangle(
                    frame,
                    (box[0], box[1]),
                    (box[0] + box[2], box[1] + box[3]),
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                # Draw the centroid for visualization
                centroid = get_centroid(box)
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

                # For tracking and line counting, use only vehicles
                if label in vehicle_classes:
                    vehicle_detections.append((box, label))

    # Update trackers with current vehicle detections
    trackers, next_object_id = update_trackers(
        vehicle_detections, trackers, next_object_id
    )

    # Check each tracked vehicle to see if it has crossed the counting line.
    # Here, we assume vehicles are moving downward (i.e. increasing y-coordinate).
    for obj_id, (centroid, counted) in trackers.items():
        cx, cy = centroid
        if not counted and cy > line_y:
            line_count += 1
            trackers[obj_id] = (centroid, True)  # Mark this object as counted

    # Draw the counting line
    cv2.line(frame, (0, line_y), (frame_width, line_y), line_color, line_thickness)
    # Display the current count on the frame
    cv2.putText(
        frame,
        f"Vehicles Passed: {line_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Intersection Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Exit when ESC is pressed
        break

cap.release()
cv2.destroyAllWindows()
