import cv2
import time

from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
from ultralytics import solutions

#from CarDetection import framecount

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "Cashmere.MP4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("YOLO11 Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO11 Tracking", 1920,1080)
vboxsize = 3
vboxes = False
classes = [0 , 2]

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_counter = 0
# Define region points
# region_points = [(485, 250), (677, 205), (855, 755), (1550, 700)]
region_points = [(485, 250), (670, 205), (1550,700), (840,755)]

# Define a counting line (here, a horizontal line at 50% of frame height)
line_y = int(frame_height * 0.3)
line_color = (0, 255, 255)  # Yellow line
line_thickness = 3
line_count = []
line_on = False
traffic_light = False


# Init TrackZone (Object Tracking in Zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",  # You can use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
    # line_width=1,  # Adjust the line width for bounding boxes and text display
    # classes=[0, 2],  # If you want to count specific classes i.e. person and car with COCO pretrained model.
)

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    classes=[0, 1, 2, 3, 5, 7],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust the line width for bounding boxes and text display
)

# ------------------------
# Initialize trackers, counters, and timer
# ------------------------
# Using a simple centroid-based tracker:
# Format: {object_id: (centroid, counted_flag)}
trackers = {}
next_object_id = 0
car_counter = 0

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
        min_dist = float('inf')
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
        print(best_id, counted_flag)
    return updated_trackers, next_object_id


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    #print(frame_counter)
    frame_counter+= 1

    if success:

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes = [0, 1, 2, 3, 5, 7])
        #print(results)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot(line_width=vboxsize,conf=False,boxes=vboxes)
        frame = results[0].plot(line_width=vboxsize, conf=False, boxes=vboxes)
        # Calculate the countdown timer value
        elapsed = time.time() - last_reset_time
        countdown = max(0, timer_duration - elapsed)

        # Display the timer and vehicle count on the frame
        cv2.putText(frame, f"Timer: {int(countdown)}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Vehicles Passed: {line_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if line_on:
            cv2.line(frame, (550, line_y + 40), (1200, line_y), line_color, line_thickness)

        if traffic_light:
            # Traffic light dimensions and position
            x, y, w, h = 50, 50, 80, 200  # x, y coordinates, width, height
            padding = 5
            circle_radius = (w - 2 * padding) // 2

            # Draw the black rectangle for the traffic light
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

            # Draw the red light
            cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, (0, 0, 255), -1)

            # Draw the yellow light
            cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, (0, 255, 255), -1)

            # Draw the green light
            cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, (0, 255, 0), -1)

        # Display the annotated frame
        #cv2.imshow("YOLO11 Tracking", annotated_frame)
        cv2.imshow("YOLO11 Tracking", frame)
        #print(car_counter)

        #frame = trackzone.trackzone(annotated_frame)
        #frame = counter.count(annotated_frame)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord("1"):
            vboxsize = 1
        elif key == ord("2"):
            vboxsize = 2
        elif key == ord("b"):
            if vboxes == True:
                vboxes = False
            else: vboxes = True

        elif key == ord("l"):
            region_points = [(550, line_y + 40), (1200, line_y)]
            if line_on == True:
                line_on = False
            else: line_on = True

        elif key == ord("t"):
            if traffic_light == True:
                traffic_light = False
            else: traffic_light = True

        elif key == ord("q"):
            break
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
        #print(vboxsize)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()