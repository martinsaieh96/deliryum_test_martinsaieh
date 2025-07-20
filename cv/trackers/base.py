from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections):
        """Recibe una lista de bboxes [[x1, y1, x2, y2, conf], ...] y retorna [[x1, y1, x2, y2, id], ...]"""
        pass

