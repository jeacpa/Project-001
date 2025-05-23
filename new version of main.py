import cv2
import threading
import base64
import flet as ft
import numpy as np  # Import numpy for frame handling
from ultralytics import YOLO
from drawing import (
    process_video,
)  # Import the process_video function from the drawing module


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

        self.video_image = ft.Image(
            border_radius=ft.border_radius.all(0), src_base64=""
        )

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
                        ),
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
                                icon_color="Green500",
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
        self.content = ft.Column(controls=[self.display_view, self.edit_view])

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
                    self.video_image.src_base64 = base64.b64encode(frame_bytes).decode(
                        "utf-8"
                    )
                    self.update()  # Update the Flet interface

                    # Display the current frame in an OpenCV window
                    frame = cv2.imdecode(
                        np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
                    )
                    cv2.imshow("Video Stream", frame)  # Show frame in OpenCV window

                    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
                        break  # Exit the loop if 'q' is pressed
                else:
                    print("frame_bytes is not of type bytes.")

        threading.Thread(
            target=capture_video, daemon=True
        ).start()  # Daemon thread to ensure it closes with the app


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
    page.fonts = {"Quicksand": "C:/Users/vbege/Downloads/Quicksand[wght].ttf"}

    todo_app = TodoApp()

    # Predefined task with embedded video URL
    predefined_task_name = "Traffic Feed"
    predefined_video_url = "https://162-d2.divas.cloud/CHAN-9749/CHAN-9749_1.stream/playlist.m3u8?66.229.161.11&vdswztokenhash=Ef1Uwba8D796xl7VOHW2Bdp4Wv2Zo1uXplNxOMB-lbs="
    todo_app.add_clicked(predefined_task_name, predefined_video_url)

    def fab_pressed(e):
        def on_submit(e):
            task_name = task_field.value
            video_url = video_field.value
            todo_app.add_clicked(task_name, video_url)
            dialog.open = False
            page.update()

        task_field = ft.TextField(label="Street", border_color="Green500")
        video_field = ft.TextField(label="URL to live feed", border_color="Green500")
        dialog = ft.AlertDialog(
            title=ft.Text("Enter Street and Live Feed URL"),
            content=ft.Column(
                [task_field, video_field], alignment=ft.MainAxisAlignment.CENTER
            ),
            actions=[
                ft.TextButton(
                    on_click=on_submit, icon="ARROW_FORWARD", icon_color="green400"
                )
            ],
            alignment=ft.alignment.center,
        )
        page.overlay.append(dialog)
        dialog.open = True
        page.update()

    search_bar = ft.SearchBar(
        view_elevation=4,
        divider_color=ft.colors.AMBER,
        bar_hint_text="Search Street...",
        view_hint_text="Choose a task from the suggestions...",
        on_change=lambda e: todo_app.search_tasks(e.data),
    )

    tabs_theme = ft.Tabs(
        overlay_color=ft.colors.with_opacity(0.1, "Green500"),
        label_color="Green500",
        indicator_color="Green500",
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Dashboard",
                content=ft.Column(
                    controls=[
                        ft.Container(
                            content=ft.Text(
                                value="traffix",
                                style=ft.TextStyle(
                                    font_family="Quicksand",
                                    size=60,
                                    weight=ft.FontWeight.BOLD,
                                ),
                            ),
                            alignment=ft.alignment.center,
                        ),
                        ft.Container(
                            content=search_bar,
                            alignment=ft.alignment.top_center,
                        ),
                        todo_app,
                    ]
                ),
            ),
            ft.Tab(
                text="Analytics",
                content=ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ),
        ],
        expand=1,
    )

    page.floating_action_button = ft.FloatingActionButton(
        icon=ft.icons.ADD, on_click=fab_pressed, bgcolor=ft.colors.GREEN_500
    )
    page.add(tabs_theme)


ft.app(main)
