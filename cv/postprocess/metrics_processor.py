import os
import json
import numpy as np

class MetricsProcessor:
    def __init__(self, persons_json_path, fps=30, activity_thresh=2.0):
        self.persons_json_path = persons_json_path
        self.fps = fps
        self.activity_thresh = activity_thresh

    def compute_metrics(self):
        with open(self.persons_json_path, "r") as f:
            data = json.load(f)
        summary = {}
        for tid, pdata in data.items():
            frames = pdata.get("frames", [])
            velocities = pdata.get("velocities", [])
            states = pdata.get("states", [])
            bboxes = pdata.get("bboxes", [])
            faces = pdata.get("faces", [])
            bodies = pdata.get("bodies", [])

            centroids = [((x1 + x2)//2, (y1 + y2)//2) for x1, y1, x2, y2 in bboxes]

            total_frames = len(frames)
            total_seconds = total_frames / self.fps if self.fps else total_frames

            mean_velocity = float(np.mean(velocities)) if velocities else 0.0

            active_frames = sum([s == "activo" for s in states])
            inactive_frames = sum([s == "inactivo" for s in states])
            active_time = active_frames / self.fps if self.fps else active_frames
            inactive_time = inactive_frames / self.fps if self.fps else inactive_frames

            summary[tid] = {
                "frames": frames,
                "centroids": centroids,
                "velocities": velocities,
                "states": states,
                "faces": faces,
                "bodies": bodies,
                "mean_velocity": mean_velocity,
                "total_time": total_seconds,
                "active_time": active_time,
                "inactive_time": inactive_time,
                "n_appearances": total_frames
            }
        return summary
