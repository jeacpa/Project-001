import os
from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt")

    dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(dir, 'vehicles.v2i.yolov11', 'data.yaml')

    trainresults = model.train(
        data = datapath,
        epochs = 5,
        # imgsz = 640,
        device = 'cpu'
    )

    # Change

    print("Training complete. Results:")
if __name__ == '__main__':

    train()