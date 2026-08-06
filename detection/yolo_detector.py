"""YOLOv8 tabanli nesne tespit katmani."""
from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: tuple  # (x1, y1, x2, y2)
    conf: float
    cls_id: int
    cls_name: str


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf_threshold: float = 0.35, classes=None):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.classes = classes

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            detections.append(Detection((x1, y1, x2, y2), conf, cls_id, cls_name))
        return detections
