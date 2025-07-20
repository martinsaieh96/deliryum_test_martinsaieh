# postprocess/metrics_processor.py

import json
import numpy as np

class MetricsProcessor:
    def __init__(self, frames_json_path: str, fps: int = 30, activity_thresh: float = 1.0):
        self.fps = fps
        self.activity_thresh = activity_thresh
        with open(frames_json_path, "r") as f:
            self.data = json.load(f)
        self.persons = {}

    def group_tracks(self):
        for frame_entry in self.data:
            frame_idx = frame_entry["frame"]
            for obj in frame_entry["objects"]:
                tid = obj["track_id"]
                bbox = obj["bbox"]
                detected = obj["detection"]
                if tid not in self.persons:
                    self.persons[tid] = []
                self.persons[tid].append({
                    "frame": frame_idx,
                    "bbox": bbox,
                    "detected": detected
                })

    def compute_metrics(self):
        self.group_tracks()
        results = {}
        for tid, tracks in self.persons.items():
            frames = [t["frame"] for t in tracks]
            bboxes = np.array([t["bbox"] for t in tracks])
            centers = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes])
            speeds = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            speeds_per_sec = speeds * self.fps
            avg_speed = float(np.mean(speeds_per_sec)) if len(speeds_per_sec) else 0
            active = speeds_per_sec > self.activity_thresh
            active_frames = int(np.sum(active))
            inactive_frames = len(frames) - active_frames
            total_secs = len(frames) / self.fps

            results[tid] = {
                "track_id": tid,
                "avg_speed_px_s": avg_speed,
                "active_frames": active_frames,
                "inactive_frames": inactive_frames,
                "total_time_secs": total_secs,
                "frames": frames,
                "trajectory": centers.tolist(),
                "avg_bbox": np.mean(bboxes, axis=0).tolist() if len(bboxes) else None,
            }
        return results
