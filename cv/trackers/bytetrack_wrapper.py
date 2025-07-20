from bytetrack import BYTETracker
import numpy as np
from .base import BaseTracker

class ByteTrackTracker(BaseTracker):
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30):
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer
        )
        self.frame_id = 0

    def update(self, detections):
        # detections: [[x1, y1, x2, y2, conf], ...]
        dets = np.array(detections)
        if dets.shape[0] == 0:
            dets = np.empty((0, 5))
        # BYTETracker espera [x1, y1, x2, y2, score]
        online_targets = self.tracker.update(dets, [], frame_id=self.frame_id)
        self.frame_id += 1
        results = []
        for t in online_targets:
            # t.tlwh: [x1, y1, w, h], t.track_id
            x1, y1, w, h = t.tlwh
            x2, y2 = x1 + w, y1 + h
            results.append([x1, y1, x2, y2, t.track_id])
        return results
