import cv2

from ultralytics import YOLO


frame_counter = 0
# Define region points
# region_points = [(485, 250), (677, 205), (855, 755), (1550, 700)]
region_points = [(485, 250), (670, 205), (1550, 700), (840, 755)]

# Define a counting line (here, a horizontal line at 50% of frame height)

line_color = (0, 255, 255)  # Yellow line


def main(video_path, line_position):
    # Load the YOLOv5 model
    model = YOLO("yolo11n.pt")  # use 'cuda' for GPU

    # Open the video
    cap = cv2.VideoCapture("Cashmere.MP4")
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    car_counter = 0
    tracked_cars = {}  # Dictionary to track cars across frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(frame_height)
        # line_y = int(frame_height * 0.3)
        # Make detections
        results = model.predict(frame)
        detections = results.pandas().xyxy[0]  # xyxy format for bounding boxes

        # Draw a line on the frame
        cv2.line(
            frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2
        )

        current_frame_cars = []

        # Process detections
        for index, row in detections.iterrows():
            if row["name"] in ["car", "truck"]:  # Consider 'car' and 'truck' classes
                bbox_center = int(row["ymin"] + (row["ymax"] - row["ymin"]) / 2)

                # Track each car by a unique ID using its bounding box and class
                car_id = (
                    int(row["xmin"]),
                    int(row["ymin"]),
                    int(row["xmax"]),
                    int(row["ymax"]),
                )

                # Check if the car has already crossed the line and has been tracked
                if car_id in tracked_cars:
                    continue

                # Check if the car crosses the line in the current frame
                if bbox_center > line_position:
                    current_frame_cars.append(car_id)
                    car_counter += 1

                # Draw bounding boxes and labels
                cv2.rectangle(
                    frame,
                    (int(row["xmin"]), int(row["ymin"])),
                    (int(row["xmax"]), int(row["ymax"])),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{row['name']} {row['confidence']:.2f}",
                    (int(row["xmin"]), int(row["ymin"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # Update tracked cars list
        tracked_cars.update({car: True for car in current_frame_cars})

        # Display count
        cv2.putText(
            frame,
            f"Vehicles Counted: {car_counter}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
