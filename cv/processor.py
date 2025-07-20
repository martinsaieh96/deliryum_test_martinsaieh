import yaml
import cv2
import os
from tqdm import tqdm
from cv.detector import PersonDetector
from cv.trackers.tracker_factory import create_tracker
from cv.face_detectors.factory import create_face_detector
from cv.face_recognizer import TopFacesManager
from cv.data_writer import DataWriter
from cv.visualizer import draw_boxes
from cv.utils import ensure_dir, save_crop, get_crop_from_bbox, compute_iou, save_face_crop

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
        self.faces_manager = TopFacesManager(max_faces=3)

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
            detected, conf = self._match_detection(bbox, detections)
            crop = get_crop_from_bbox(frame, bbox)
            crop_path = save_crop(crop, self.crops_dir, tid, frame_idx)
            face_path = None
            faces = self.face_detector.detect(crop)
            if faces:
                best = max(faces, key=lambda f: f["conf"] * (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                face_img = crop[best["bbox"][1]:best["bbox"][3], best["bbox"][0]:best["bbox"][2]]
                self.faces_manager.consider(tid, face_img, frame_idx, score=best["conf"])
                face_path = save_face_crop(face_img, self.faces_dir, tid, frame_idx)

            track_infos.append({
                "track_id": tid,
                "bbox": bbox,
                "detection": detected,
                "confidence": conf,
                "crop_path": crop_path,
            })
        self.data_writer.add_frame(frame_idx, track_infos)
        frame_annotated = draw_boxes(frame, tracks)
        out_vid.write(frame_annotated)

    def _match_detection(self, bbox, detections, iou_thresh=0.5):
        for d in detections:
            iou = compute_iou(bbox, d['bbox'])
            if iou > iou_thresh:
                return True, d['conf']
        return False, 0

    def finalize(self):
        face_paths_dict = self.faces_manager.save_faces(self.faces_dir)
        self.data_writer.save()
