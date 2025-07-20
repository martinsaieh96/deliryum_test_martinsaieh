import cv2
import os
from .base import BaseFaceDetector

class HaarCascadeDetector(BaseFaceDetector):
    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=[56, 56]):
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(haar_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = tuple(minSize)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        faces = []
        for (x, y, w, h) in boxes:
            faces.append({
                "bbox": [x, y, x+w, y+h],
                "conf": 1.0  # Haarcascade no da score
            })
        return faces
