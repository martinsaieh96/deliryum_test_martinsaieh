import face_recognition
from .base import BaseFaceDetector

class FaceRecogDetector(BaseFaceDetector):
    def __init__(self, model="hog"):
        self.model = model

    def detect(self, image):
        rgb_img = image[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb_img, model=self.model)
        faces = []
        for box in boxes:
            top, right, bottom, left = box
            faces.append({
                "bbox": [left, top, right, bottom],
                "conf": 1.0  # No da score, asumimos 1.0
            })
        return faces
