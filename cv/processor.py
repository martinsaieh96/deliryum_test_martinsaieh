import yaml
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from cv.detector import PersonDetector
from cv.trackers.tracker_factory import create_tracker
from cv.face_detectors.factory import create_face_detector
from cv.face_recognizer import TopFacesManager
from cv.data_writer import DataWriter
from cv.visualizer import draw_boxes
from cv.utils import (ensure_dir, save_crop, get_crop_from_bbox,get_expanded_bbox,
                    compute_iou, save_face_crop, make_gallery_per_person)

class VideoProcessor:
    def __init__(self, config_path, video_name):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.video_name = video_name
        self._setup()

    def _setup(self):
        c = self.cfg
        self.detector = PersonDetector(c['detector']['model_path'], c['detector']['min_bbox_area'])
        self.tracker = create_tracker(c['tracker_active'], c['trackers'][c['tracker_active']])
        self.face_detector = create_face_detector(c)
        self.faces_manager = TopFacesManager()

        base_name = os.path.splitext(self.video_name)[0]
        self.data_writer = DataWriter(os.path.join(c['paths']['json'], f"frames_{self.video_name}.json"), mode="json")

        self.crops_dir = os.path.join(c['paths']['crops'], base_name)
        self.faces_dir = os.path.join(c['paths']['faces'], base_name)
        self.video_path = os.path.join(c['paths']['raw_videos'], self.video_name)
        self.save_video_path = os.path.join(c['paths']['processed_videos'], f"output_{base_name}.mp4")

        ensure_dir(self.crops_dir)
        ensure_dir(self.faces_dir)
        ensure_dir(c['paths']['processed_videos'])
        ensure_dir(c['paths']['json'])

        self.person_data = defaultdict(lambda: {
            "bboxes": [],
            "frames": [],
            "centroids": [],
            "velocities": [],
            "states": [],
        })

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid = cv2.VideoWriter(self.save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_idx = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, frame_idx, out_vid)
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        out_vid.release()
        self.finalize()
        print("Procesamiento completado.")

    def process_frame(self, frame, frame_idx, out_vid):
        detections = self.detector.detect(frame)
        dets_for_tracker = [d['bbox'] + [d['conf']] for d in detections]
        tracks = self.tracker.update(dets_for_tracker)

        track_infos = []
        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)
            bbox = [x1, y1, x2, y2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.person_data[tid]["bboxes"].append(bbox)
            self.person_data[tid]["frames"].append(frame_idx)
            self.person_data[tid]["centroids"].append((cx, cy))
            detected, conf = self._match_detection(bbox, detections)
            crop = get_crop_from_bbox(frame, bbox)
            crop_path = save_crop(crop, self.crops_dir, tid, frame_idx)
            face_path = None
        faces = self.face_detector.detect(crop)
        if faces:
            # Elige la mejor cara
            best = max(
                faces,
                key=lambda f: f["conf"] * (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
            )
            rel_face_bbox = best.get("bbox", None)
            if rel_face_bbox is not None and len(rel_face_bbox) == 4:
                abs_face_bbox = [
                    x1 + rel_face_bbox[0],
                    y1 + rel_face_bbox[1],
                    x1 + rel_face_bbox[2],
                    y1 + rel_face_bbox[3]
                ]
                expanded_bbox = get_expanded_bbox(abs_face_bbox, frame.shape, scale=1.3)
                fx1, fy1, fx2, fy2 = expanded_bbox
                face_img = frame[fy1:fy2, fx1:fx2]

                self.faces_manager.consider(tid, face_img, frame_idx, score=best["conf"], bbox_cuerpo=bbox)
                face_path = save_face_crop(face_img, self.faces_dir, tid, frame_idx)

            track_infos.append({
                "track_id": tid,
                "bbox": bbox,
                "detection": detected,
                "confidence": conf,
                "crop_path": crop_path,
            })
        self.data_writer.add_frame(frame_idx, track_infos)
        frame_annotated = draw_boxes(frame, tracks, self.person_data, frame_idx)
        out_vid.write(frame_annotated)

    def _match_detection(self, bbox, detections, iou_thresh=0.5):
        for d in detections:
            iou = compute_iou(bbox, d['bbox'])
            if iou > iou_thresh:
                return True, d['conf']
        return False, 0

    def finalize(self):
        top_faces_dir = os.path.join(self.cfg['paths']['top_faces'], os.path.splitext(self.video_name)[0])
        top_bodies_dir = os.path.join(self.cfg['paths']['top_bodies'], os.path.splitext(self.video_name)[0])
        self.faces_manager.save_faces(top_faces_dir, top_bodies_dir, self.video_path)
        gallery_dir = os.path.join(self.cfg['paths']['gallery'], os.path.splitext(self.video_name)[0])
        make_gallery_per_person(top_faces_dir, gallery_dir)

        self.data_writer.save()  
        output_json = os.path.join(self.cfg['paths']['json'], f"persons_{self.video_name}.json")
        self.data_writer.save_per_person(output_json, top_faces_dir=top_faces_dir, top_bodies_dir=top_bodies_dir, max_faces=8, max_bodies=3)

