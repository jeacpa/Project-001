import cv2
import cvzone
import math
from ultralytics import YOLO
import numpy as np
import base64

# List to store points
points = []
shapes = []
current_line_color = (255, 0, 0)  # Initial line color (red)
modifications_allowed = True  # Flag to allow or disallow modifications
frame_copy = None  # Initialize frame_copy
mouse_pos = None  # Store current mouse position

# Lists to store vehicle counts over time
green_counts = []
blue_counts = []
red_counts = []
orange_counts = []
purple_counts = []

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
classnames = model.names

# Define the classes to detect
target_classes = ['car', 'truck', 'motorcycle', 'bicycle', 'bus']

def draw_points(event, x, y, flags, param):
    global points, frame_copy, current_line_color, modifications_allowed, mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN and modifications_allowed:
        points.append((x, y))

        if len(points) > 1:
            cv2.line(frame_copy, points[-2], points[-1], current_line_color, 2, lineType=cv2.LINE_AA)

    # Update the mouse position when the mouse is moved
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

def point_line_distance(p1, p2, p):
    num = abs((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0])
    den = math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return num / den if den != 0 else float('inf')  # Return infinite if the line is a point

def delete_shape(x, y):
    global shapes
    for i, (shape, color) in enumerate(shapes):
        if len(shape) > 2:
            for j in range(len(shape) - 1):
                # Check if the point (x, y) is near the line segment
                dist = point_line_distance(shape[j], shape[j + 1], (x, y))
                if dist < 5:  # Allow a small distance tolerance
                    del shapes[i]  # Delete the shape
                    return  # Exit after deleting one shape


def draw_smooth_dotted_line(frame, start_point, end_point, color, dot_radius=4, dot_spacing=15):
    """Function to draw a smoother dotted line between two points."""
    dist = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    num_dots = int(dist // dot_spacing)

    for i in range(num_dots):
        # Calculate the center of each dot
        dot_center = (
            int(start_point[0] + (end_point[0] - start_point[0]) * (i / num_dots)),
            int(start_point[1] + (end_point[1] - start_point[1]) * (i / num_dots)),
        )
        # Draw a filled circle to make a smoother dot
        cv2.circle(frame, dot_center, dot_radius, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)  # Anti-aliased


def process_video(video_url):
    global frame_copy, points, shapes, current_line_color, modifications_allowed, mouse_pos
    cap = cv2.VideoCapture(video_url)

    # Skip the first 1800 frames
    for _ in range(1800):
        cap.read()

    # Create a window and set the mouse callback function
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", draw_points)

    while cap.isOpened():
        detections = np.empty((0, 5))
        car_count = 0
        green_shape_count = 0
        blue_shape_count = 0
        red_shape_count = 0
        orange_shape_count = 0
        purple_shape_count = 0

        ret, video = cap.read()
        if not ret:
            break

        video = cv2.resize(video, (1600, 830))
        frame_copy = video.copy()

        results = model.track(video, persist=True)

        for result in results:
            boxes = result.boxes
            # Count vehicles based on shape colors
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                cv2.circle(video, (cx, cy), 6, (0, 255, 255), -1)
                cvzone.cornerRect(video, (x1, y1, w, h), l=10, rt=0, colorR=(255, 0, 255))

                # Count vehicles based on shape colors
                for shape, color in shapes:
                    if len(shape) > 2:
                        counts = cv2.pointPolygonTest(np.array(shape, np.int32), pt=(cx, cy), measureDist=False)
                        if counts >= 0:  # Inside the defined region
                            if color == (0, 255, 0):  # Green shape
                                green_shape_count += 1
                            elif color == (255, 0, 0):  # Blue shape
                                blue_shape_count += 1
                            elif color == (0, 0, 255):  # Red shape
                                red_shape_count += 1
                            elif color == (0, 165, 255):  # Orange shape
                                orange_shape_count += 1
                            elif color == (128, 0, 128):  # Purple shape
                                purple_shape_count += 1

            # Calculate the centroid of each shape and display the counter at the centroid
            # Calculate the centroid of each shape and display the counter at the centroid
            for shape, color in shapes:
                if len(shape) > 2:
                    M = cv2.moments(np.array(shape, np.int32))
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        text = ""
                        # Determine the counter text based on the shape color
                        if color == (0, 255, 0):  # Green shape
                            text = f'{green_shape_count}'
                            text_color = (0, 255, 0)  # Green color
                        elif color == (255, 0, 0):  # Blue shape
                            text = f'{blue_shape_count}'
                            text_color = (255, 0, 0)  # Blue color
                        elif color == (0, 0, 255):  # Red shape
                            text = f'{red_shape_count}'
                            text_color = (0, 0, 255)  # Red color
                        elif color == (0, 165, 255):  # Orange shape
                            text = f'{orange_shape_count}'
                            text_color = (0, 165, 255)  # Orange color
                        elif color == (128, 0, 128):  # Purple shape
                            text = f'{purple_shape_count}'
                            text_color = (128, 0, 128)  # Purple color

                        # Get the text size
                        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        # Draw a white rectangle behind the text
                        cv2.rectangle(video, (cX - w // 2 - 5, cY - h // 2 - 5), (cX + w // 2 + 5, cY + h // 2 + 5),
                                      (255, 255, 255), -1)
                        # Put the text on the video frame with the corresponding shape color
                        cv2.putText(video, text, (cX - w // 2, cY + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Draw current points and lines
        for i, point in enumerate(points):
            if i > 0:
                cv2.line(video, points[i - 1], points[i], current_line_color, 2, lineType=cv2.LINE_AA)

        # Draw saved shapes
        for shape, color in shapes:
            if len(shape) > 2:
                for i in range(len(shape) - 1):
                    cv2.line(video, shape[i], shape[i + 1], color, 1, lineType=cv2.LINE_AA)

        # Draw a dashed line from the last point to the current mouse position
        if mouse_pos and len(points) > 0:
            draw_smooth_dotted_line(video, points[-1], mouse_pos, current_line_color)

        # Convert frame to base64 for Flet
        _, buffer = cv2.imencode('.jpg', video)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Use a placeholder to return the current frame
        yield frame_base64

        cv2.imshow("Video", video)  # Display video in OpenCV

        # Check for key presses
        k = cv2.waitKey(1) & 0xFF
        if k == ord('u'):  # 'u' key to disallow further modifications
            modifications_allowed = False
        elif k == ord('e'):  # 'e' key to allow further modifications
            modifications_allowed = True

        # Handle key presses for drawing and managing shapes
        if modifications_allowed:
            if k == 13:  # Enter key
                if len(points) > 1:
                    shapes.append((points.copy(), current_line_color))
                points = []
            elif k == ord('d') and points:  # 'd' key for undo
                points.pop()
            elif k == ord(' '):  # Space key
                if len(points) > 1:
                    shapes.append((points.copy(), current_line_color))
                points = []
            elif k == ord('l'):  # 'l' key to change current line color to green
                current_line_color = (0, 255, 0)
            elif k == ord('r'):  # 'r' key to change current line color to red
                current_line_color = (0, 0, 255)
            elif k == ord('o'):  # 'o' key to change current line color to orange
                current_line_color = (0, 165, 255)
            elif k == ord('p'):  # 'p' key to change current line color to purple
                current_line_color = (128, 0, 128)
            elif k == ord('t'):  # 't' key to delete the shape at clicked position
                if points:
                    last_x, last_y = points[-1]  # Replace with the last clicked point
                    delete_shape(last_x, last_y)

    cap.release()
    cv2.destroyAllWindows()
