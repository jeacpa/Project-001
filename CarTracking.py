from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# results = model.track(source='Cashmere.MP', show=True, tracker = "bytetrack.yaml")
results = model.track(source="Cashmere.MP", show=True)
