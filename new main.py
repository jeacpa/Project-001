import cv2
import threading
import base64
import flet as ft
import numpy as np  # Import numpy for frame handling
from ultralytics import YOLO
from drawing import process_video  # Import the process_video function from the drawing module

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
classnames = model.names


class Task(ft.Container):
    def __init__(self, task_name, video_url, task_delete):
        super().__init__()
        self.task_name = task_name
        self.video_url = video_url
        self.task_delete = task_delete
        self.edit_name = ft.TextField(expand=1, multiline=True, border_color="Green500")

        self.video_image = ft.Image(border_radius=ft.border_radius.all(0), src_base64="")

        self.display_view = ft.Container(
            bgcolor=ft.colors.WHITE,
            padding=0,
            border_radius=20,
            content=ft.Column(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=0,
                controls=[
                    self.video_image,
                    ft.Container(
                        content=ft.Text(
                            value=self.task_name,
                            color=ft.colors.BLACK,
                            style=ft.TextStyle(
                                font_family="Quicksand",  # Ensure the custom font is used
                                size=30,  # Set the desired font size here
                                weight=ft.FontWeight.NORMAL,
                            ),
                        )
                        ,
                        alignment=ft.alignment.center,
                        padding=0,
                        margin=0,
                    ),
                    ft.Row(
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=10,
                        controls=[
                            ft.IconButton(
                                icon=ft.icons.CREATE,
                                tooltip="Edit",
                                on_click=self.edit_clicked,
                                icon_color="Green500"

                            ),
                            ft.IconButton(
                                ft.icons.DELETE,
                                tooltip="Delete",
                                on_click=self.delete_clicked,
                                icon_color="Green500",
                            ),
                        ],
                    ),
                ],
            ),
        )

        self.edit_view = ft.Row(
            visible=False,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            spacing=5,
            controls=[
                self.edit_name,
                ft.IconButton(
                    icon=ft.icons.DONE_OUTLINE_OUTLINED,
                    icon_color=ft.colors.GREEN,
                    tooltip="Update To-Do",
                    on_click=self.save_clicked,
                ),
            ],
        )
        self.content = ft.Column(
            controls=[self.display_view, self.edit_view]
        )

    def edit_clicked(self, e):
        self.edit_name.value = self.task_name
        self.display_view.visible = False
        self.edit_view.visible = True
        self.update()

    def save_clicked(self, e):
        self.task_name = self.edit_name.value
        self.display_view.content.controls[1].content.value = self.task_name
        self.display_view.visible = True
        self.edit_view.visible = False
        self.update()

    def delete_clicked(self, e):
        self.task_delete(self)

    def start_video_capture(self):
        def capture_video():
            for frame_bytes in process_video(self.video_url):
                # Check if the frame_bytes is a base64 string
                if isinstance(frame_bytes, str):
                    # Decode the base64 string to bytes
                    try:
                        frame_bytes = base64.b64decode(frame_bytes)
                    except Exception as e:
                        print(f"Error decoding base64 string: {e}")
                        continue  # Skip this frame if decoding fails

                # Ensure that frame_bytes is of type bytes before encoding
                if isinstance(frame_bytes, bytes):
                    # Update the Flet interface
                    self.video_image.src_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                    self.update()  # Update the Flet interface

                    # Display the current frame in an OpenCV window
                    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow("Video Stream", frame)  # Show frame in OpenCV window

                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                        break  # Exit the loop if 'q' is pressed
                else:
                    print("frame_bytes is not of type bytes.")

        threading.Thread(target=capture_video, daemon=True).start()  # Daemon thread to ensure it closes with the app


class TodoApp(ft.Column):
    def __init__(self):
        super().__init__()
        self.tasks = ft.GridView(
            expand=True,
            runs_count=2,
            max_extent=500,
            child_aspect_ratio=1.0,
            spacing=10,
            run_spacing=10,

        )

        self.controls = [
            ft.Row(
                [ft.Text(value="", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM)],
                alignment=ft.MainAxisAlignment.CENTER,
            ),

            self.tasks,
        ]

    def add_clicked(self, task_name, video_url):
        if task_name and video_url:
            task = Task(task_name, video_url, self.task_delete)
            self.tasks.controls.append(task)
            if self.page:
                self.update()
            task.start_video_capture()

    def task_delete(self, task):
        self.tasks.controls.remove(task)
        if self.page:
            self.update()

    def search_tasks(self, query):
        for task in self.tasks.controls:
            task.visible = query.lower() in task.task_name.lower()
        if self.page:
            self.update()


def main(page: ft.Page):
    page.title = "ToDo App"
    page.scroll = "auto"

    # Add custom fonts
    page.fonts = {
        "Quicksand": "C:/Users/vbege/Downloads/Quicksand[wght].ttf"
    }
