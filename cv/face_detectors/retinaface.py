import numpy as np
import cv2
from .base import BaseFaceDetector
from insightface.app import FaceAnalysis

class RetinaFaceDetector(BaseFaceDetector):
    def __init__(self, threshold=0.9, det_size=(640, 640), ctx_id=0):
        self.threshold = threshold
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._ref_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        self._init_detector()

    def _init_detector(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        print("[INFO] Providers:", self.app.models['detection'].session.get_providers())
    def _align_face(self, image, landmarks, output_size=(112, 112)):
        landmarks = np.array(landmarks).astype(np.float32)
        if landmarks.shape != (5, 2):
            raise ValueError(f"Se esperaban 5 landmarks, se obtuvo: {landmarks.shape}")
        tform = cv2.estimateAffinePartial2D(landmarks, self._ref_landmarks, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(image, tform, output_size, borderValue=0.0)
        return aligned

    def detect(self, image):
        if image.shape[-1] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image

        faces = []
        results = self.app.get(img_rgb)

        for res in results:
            score = getattr(res, 'det_score', 1.0)
            if score < self.threshold:
                continue

            bbox = res.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            landmarks = res.landmark_2d_5.astype(np.float32)  # (5, 2)

            face_data = {
                "bbox": bbox,
                "conf": float(score),
                "landmarks": landmarks,
            }
            face_data["aligned"] = 'a'#res.crop_affine  
            faces.append(face_data)
        return faces

