from enum import Enum
import cv2
import time

from sympy.printing.pretty.pretty_symbology import annotated
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics import solutions
import tkinter as tk

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class VideoReadException(Exception):
    pass

class LightColor(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3

YELLOW_LIGHT = (0, 255, 255)
GREEN_LIGHT = (0, 255, 0)
BLACK_LIGHT = (65, 74, 76)
RED_LIGHT = (0, 0, 255)

TRACKING_CLASSES = [1, 2, 3, 5, 7]
LINE_COLOR = (0, 255, 255)  # Yellow line
LINE_THICKNESS = 3

BOX_LINE_COLOR = (0, 165, 255)
BOX_TEXT_COLOR = (0, 165, 255)
BOX_LINE_THICKNESS = 2
BOX_TEXT_THICKNESS = 1
BOX_TEXT_SCALE = 0.4

class Expiriment:
    model: YOLO
    video_path: str
    cap: cv2.VideoCapture
    frame_counter: int
    show_text: bool
    show_boxes: bool
    show_zones: bool
    show_light: bool
    vboxsize: int
    should_exit: bool
    window_exists: bool
    light_color: LightColor
    tracking_results: Results

    def __init__(self, video_path: str, show_boxes: bool = True, show_zones: bool = True, show_text: bool = True, show_light: bool = False):
        self.model = YOLO("yolo11n.pt")
        self.video_path = video_path
        self.model.overrides["verbose"] = False
        self.frame_counter = 0
        self.vboxsize = 1
        self.show_boxes = show_boxes
        self.show_zones = show_zones
        self.show_light = show_light
        self.show_text = show_text
        self.should_exit = False
        self.window_exists = False
        self.light_color = LightColor.RED

    def _render_zones(self, frame):
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(frame_height * 0.3)
        cv2.line(frame, (545, line_y + 30), (890, line_y), LINE_COLOR, LINE_THICKNESS)

    def _render_text(self, frame):
        cv2.putText(frame, f"Timer: ??", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Vehicles: ??", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        time_offset = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(frame, f"Time Offset: {round(time_offset/1000,2)}s", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    def _render_traffic_light(self, frame):
        x, y, w, h = 1800, 1, 80, 300  # x, y coordinates, width, height
        padding = 7
        circle_radius = (w - 2 * padding) // 2

        cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, RED_LIGHT if self.light_color == LightColor.RED else BLACK_LIGHT, -1)
        cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, YELLOW_LIGHT if self.light_color == LightColor.YELLOW else BLACK_LIGHT, -1)
        cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, GREEN_LIGHT if self.light_color == LightColor.GREEN else BLACK_LIGHT, -1)

    def _render_boxes(self, frame):
        if not self.tracking_results:
            return
        boxes = self.tracking_results.boxes.xyxy.cpu().numpy()       # Bounding boxes
        ids = self.tracking_results.boxes.id.cpu().numpy().astype(int)  # Track IDs
        classes = self.tracking_results.boxes.cls.cpu().numpy().astype(int)  # Class indices
        names = self.model.names

        for box, obj_id, cls_id in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            # label = f"{names[cls_id]} #{obj_id}"
            label = f"{obj_id}"

            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, BOX_TEXT_SCALE, BOX_TEXT_THICKNESS)
            baseline += 2

            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2            
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            box_coords1 = (text_x - 4, text_y - text_height - 4)
            box_coords2 = (text_x + text_width + 4, text_y + baseline)

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_LINE_COLOR, BOX_LINE_THICKNESS, cv2.LINE_AA)
            cv2.rectangle(frame, box_coords1, box_coords2, BOX_TEXT_COLOR, cv2.FILLED)

            # cv2.putText(frame, label, (x1+1, (y1 - 10)+1),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), BOX_TEXT_THICKNESS, cv2.LINE_AA)        
            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, BOX_TEXT_SCALE, (0,0,0), BOX_TEXT_THICKNESS, cv2.LINE_AA)        

    def _analyze_frame(self, frame):
        results = self.model.track(frame, persist=True, classes = TRACKING_CLASSES)
        self.tracking_results = results[0]
        # ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
        # print(ids)
        # boxes = results[0].boxes.xywh.cpu()

        # Let cv render boxes
        if self.show_boxes:
            self._render_boxes(frame)
            # frame = results[0].plot(line_width=self.vboxsize, conf=True, boxes=boxes)
        if self.show_zones:
            self._render_zones(frame)
        if self.show_light:
            self._render_traffic_light(frame)
        if self.show_text:
            self._render_text(frame)

        return frame

    def _check_keypress(self):

        key = cv2.waitKey(1)
        if key == ord("1"):
            self.vboxsize = 1
        elif key == ord("2"):
            self.vboxsize = 2
        elif key == ord("b"):
            self.show_boxes = not self.show_boxes
        elif key == ord("l"):
            self.show_zones = not self.show_zones
        elif key == ord("t"):
            self.show_light = not self.show_light
        elif key == ord("q"):
            self.should_exit = True

    def run_video_analysis(self):
        print("Loading video...")

        self.cap = cv2.VideoCapture(self.video_path)

        # Start 27s in to skip initial cross traffic
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 27000)

        while self.cap.isOpened() and not self.should_exit:
            # Read a frame from the video
            success, frame = self.cap.read()

            self.frame_counter += 1

            if not success:
                raise VideoReadException("Could not read from video")
            
            frame_out = self._analyze_frame(frame)

            if not self.window_exists:
                cv2.namedWindow("YOLO11 Tracking", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO11 Tracking", 1920,1080)

                self.window_exists = True

            cv2.imshow("YOLO11 Tracking", frame_out)

            self._check_keypress()

        self.cap.release()

        cv2.destroyAllWindows()

exp = Expiriment("Cashmere.MP4")
exp.run_video_analysis()

# Open the video file
# video_path = "Cashmere.MP4"
# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_MSEC, 27000)

# window_exists = False
# vboxsize = 3
# vboxes = True


# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_counter = 0
# Define region points
# region_points = [(485, 250), (677, 205), (855, 755), (1550, 700)]
# region_points = [(485, 250), (670, 205), (1550,700), (840,755)]

# Define a counting line (here, a horizontal line at 50% of frame height)
# line_y = int(frame_height * 0.3)
# line_color = (0, 255, 255)  # Yellow line
# line_thickness = 3
# line_count = []

# countdown = 10
# line_on = True
# traffic_light = True
# red_light = (0, 0, 255)
# yellow_light = (0, 255, 255)
# green_light = (0, 255, 0)
# black_light = (65, 74, 76)
# light_color = red_light


# def change_light(light_color):
#     x, y, w, h = 1800, 1, 80, 300  # x, y coordinates, width, height
#     padding = 7
#     circle_radius = (w - 2 * padding) // 2

#     # Draw the black rectangle for the traffic light
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

#     if light_color == red_light:
#         # Draw the red light
#         cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, red_light, -1)
#         cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, black_light, -1)
#         return light_color

#     elif light_color == yellow_light:
#         # Draw the yellow light
#         cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, yellow_light, -1)
#         cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, black_light, -1)
#         return light_color

#     elif light_color == green_light:
#         # Draw the green light
#         cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, green_light, -1)
#         return light_color

#     elif light_color == black_light:
#         # Draw the green light
#         cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, black_light, -1)
#         cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, black_light, -1)
#         return light_color

# Init TrackZone (Object Tracking in Zones, not complete frame)
# trackzone = solutions.TrackZone(
#     show=True,  # Display the output
#     region=region_points,  # Pass region points
#     model="yolo11n.pt",  # You can use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
#     # line_width=1,  # Adjust the line width for bounding boxes and text display
#     # classes=[0, 2],  # If you want to count specific classes i.e. person and car with COCO pretrained model.
# )

# # Init ObjectCounter
# counter = solutions.ObjectCounter(
#     show=True,  # Display the output
#     region=region_points,  # Pass region points
#     model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
#     classes=[0, 1, 2, 3, 5, 7],  # If you want to count specific classes i.e person and car with COCO pretrained model.
#     show_in=True,  # Display in counts
#     show_out=True,  # Display out counts
#     line_width=2,  # Adjust the line width for bounding boxes and text display
# )

# ------------------------
# Initialize trackers, counters, and timer
# ------------------------
# Using a simple centroid-based tracker:
# Format: {object_id: (centroid, counted_flag)}
# trackers = {}
# next_object_id = 0
# car_counter = 0

# Countdown timer settings

# timer_duration = 10  # seconds
# last_reset_time = time.time()  # resets each time a vehicle crosses the line

# def get_centroid(box):
#     """Compute centroid given a bounding box (x1, y1, x2, y2)."""
#     x1, y1, x2, y2 = box
#     return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# def update_trackers(detections, trackers, next_object_id, distance_threshold=100):
#     """
#     A simple tracker that associates current detections with existing ones
#     based on Euclidean distance. If no match is found, a new tracker is created.
#     """
#     updated_trackers = {}
#     for box, label in detections:
#         centroid = get_centroid(box)
#         best_id = None
#         min_dist = float('inf')
#         for obj_id, (prev_centroid, counted) in trackers.items():
#             dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
#             if dist < distance_threshold and dist < min_dist:
#                 best_id = obj_id
#                 min_dist = dist
#         if best_id is None:
#             best_id = next_object_id
#             next_object_id += 1
#         # Preserve the counted flag if it exists
#         counted_flag = trackers.get(best_id, (None, False))[1]
#         updated_trackers[best_id] = (centroid, counted_flag)
#         print(best_id, counted_flag)
#     return updated_trackers, next_object_id

# exp = Expirement("Cashmere.MP4")
# exp.run_video_analysis()


# Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()
#     #print(frame_counter)
#     frame_counter+= 1

#     if success:

#         # Run YOLO11 tracking on the frame, persisting tracks between frames

#         results = model.track(frame, persist=True, classes = tracking_classes)
#         ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
#         print(ids)
#         boxes = results[0].boxes.xywh.cpu()
#         # print(boxes)

#         #print(results)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot(line_width=vboxsize,conf=False,boxes=vboxes)
#         frame = results[0].plot(line_width=vboxsize, conf=True, boxes=vboxes)
#         # Calculate the countdown timer value
#         if light_color == green_light:
#             elapsed = time.time() - last_reset_time
#             countdown = max(0, timer_duration - elapsed)
#             # timer_duration = 10  # seconds
#             # last_reset_time = time.time()  # resets each time a vehicle crosses the line

#         # Display the timer and vehicle count on the frame
#         cv2.putText(frame, f"Timer: {int(countdown)}s", (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         cv2.putText(frame, f"Vehicles: {line_count}", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
#         time_offset = cap.get(cv2.CAP_PROP_POS_MSEC)
#         cv2.putText(frame, f"Time Offset: {round(time_offset)}ms", (10, 130),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         if line_on:
#             cv2.line(frame, (545, line_y + 30), (890, line_y), line_color, line_thickness)

#         if traffic_light:
#             change_light(light_color)
#             if frame_counter == 900:
#                 light_color = green_light
#                 vlight = change_light(light_color)
#                 timer_duration = 10  # seconds
#                 last_reset_time = time.time()  # resets each time a vehicle crosses the line

#             elif frame_counter == 3200:
#                 light_color = yellow_light
#                 vlight = (yellow_light)

#             elif frame_counter == 3300:
#                 light_color = red_light
#                 vlight = change_light(light_color)

#             # # Traffic light dimensions and position
#             # x, y, w, h = 1800, 1, 80, 300  # x, y coordinates, width, height
#             # padding = 7
#             # circle_radius = (w - 2 * padding) // 2
#             #
#             # # Draw the black rectangle for the traffic light
#             # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
#             #
#             # # Draw the red light
#             # cv2.circle(frame, (x + w // 2, y + h // 4), circle_radius, red_light, -1)
#             #
#             # # Draw the yellow light
#             # cv2.circle(frame, (x + w // 2, y + h // 2), circle_radius, yellow_light, -1)
#             #
#             # # Draw the green light
#             # cv2.circle(frame, (x + w // 2, y + 3 * h // 4), circle_radius, green_light, -1)

#         # Display the annotated frame
#         #cv2.imshow("YOLO11 Tracking", annotated_frame)

#         if not window_exists:
#             cv2.namedWindow("YOLO11 Tracking", cv2.WINDOW_NORMAL)
#             cv2.resizeWindow("YOLO11 Tracking", 1920,1080)

#             window_exists = True

#         cv2.imshow("YOLO11 Tracking", frame)
#         #print(car_counter)


#         # frame2 = trackzone.trackzone(annotated_frame)
#         # frame2 = counter.count(annotated_frame)
#         # qcv2.imshow("YOLO11 Trackzone", annotated_frame)

#         # Break the loop if 'q' is pressed
#         key = cv2.waitKey(1)
#         if key == ord("1"):
#             vboxsize = 1
#         elif key == ord("2"):
#             vboxsize = 2
#         elif key == ord("b"):
#             if vboxes == True:
#                 vboxes = False
#             else: vboxes = True

#         elif key == ord("l"):
#             region_points = [(550, line_y + 40), (1200, line_y)]
#             if line_on == True:
#                 line_on = False
#             else: line_on = True

#         elif key == ord("t"):
#             if traffic_light == True:

#                 vlight = change_light(light_color)
#                 traffic_light = False
#             else: traffic_light = True

#         elif key == ord("w"):
#             light_color = red_light
#             vlight = change_light(light_color)
#             timer_duration = 10  # seconds
#             elapsed = 0
#             countdown = max(0, timer_duration - elapsed)

#         elif key == ord("s"):
#             light_color = yellow_light
#             vlight = change_light(light_color)

#         elif key == ord("x"):
#             light_color = green_light
#             vlight = change_light(light_color)
#             timer_duration = 10  # seconds
#             last_reset_time = time.time()  # resets each time a vehicle crosses the line

#         elif key == ord("q"):
#             # print("q", line_y, traffic_light, frame_counter)
#             # print(centroid)
#             # print(vlight)
#             break
#         # if cv2.waitKey(1) & 0xFF == ord("q"):
#         #    break
#         #print(vboxsize)
#     else:
#         # Break the loop if the end of the video is reached
#         break

# Release the video capture object and close the display window

