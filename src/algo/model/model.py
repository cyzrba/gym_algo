from ultralytics import YOLO
from pathlib import Path


class Model:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)
    

pose_model = Model(Path(__file__).resolve().parent / "yolo26n-pose.pt")
