import os
import cv2
import json

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_crop(crop, out_dir, track_id, frame_idx):
    if crop is None or crop.size == 0:
        return None
    fname = f"{track_id}_{frame_idx}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, crop)
    return fpath

def save_face_crop(face_crop, out_dir, track_id, frame_idx):
    if face_crop is None or face_crop.size == 0:
        return None
    fname = f"{track_id}_{frame_idx}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, face_crop)
    return fpath


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_next_filename(dir_path, base, ext):
    i = 0
    while os.path.exists(os.path.join(dir_path, f"{base}_{i}.{ext}")):
        i += 1
    return os.path.join(dir_path, f"{base}_{i}.{ext}")

def get_crop_from_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    # Clip a los bordes
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    else:
        return None  # Bounding box inválido



def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou