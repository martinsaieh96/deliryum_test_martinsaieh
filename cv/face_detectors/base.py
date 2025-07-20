class BaseFaceDetector:
    def detect(self, image):
        """
        Recibe imagen BGR (numpy array).
        Retorna lista de dicts: [{'bbox': [x1, y1, x2, y2], 'conf': float}, ...]
        """
        raise NotImplementedError
