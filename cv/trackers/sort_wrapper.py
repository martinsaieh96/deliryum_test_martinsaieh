import numpy as np
from .base import BaseTracker
from .sort import Sort

class SortTracker(BaseTracker):
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections):
        dets = np.array(detections)
        if dets.shape[0] == 0:
            dets = np.empty((0, 5))
        tracked_objects = self.tracker.update(dets)
        return tracked_objects.tolist()
