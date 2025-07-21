import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path, min_bbox_area=3136):
        self.model = YOLO(model_path)
        self.min_bbox_area = min_bbox_area

    def detect(self, frame):
        results = self.model(frame)
        persons = []
        for result in results[0].boxes:
            cls = int(result.cls)
            if cls == 0:  
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area >= self.min_bbox_area:
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': float(result.conf),
                        'area': area
                    })
        return persons
