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
    print('guardando imagen')
    print(face_crop.size)
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

def make_gallery(top_faces_dir, output_path, size=(8,8), max_width=1080):
    import cv2
    import numpy as np
    from glob import glob
    import os

    face_imgs = sorted(glob(os.path.join(top_faces_dir, "tid*_face*.jpg")))
    gallery_imgs = [cv2.imread(f) for f in face_imgs if cv2.imread(f) is not None]
    gallery_imgs = [cv2.resize(img, (112, 112)) for img in gallery_imgs if img is not None]
    num_needed = size[0] * size[1]
    if len(gallery_imgs) < num_needed:
        black = np.zeros((112, 112, 3), dtype=np.uint8)
        gallery_imgs += [black] * (num_needed - len(gallery_imgs))
    gallery_imgs = gallery_imgs[:num_needed]

    rows = []
    for i in range(size[0]):
        row_imgs = gallery_imgs[i*size[1]:(i+1)*size[1]]
        row = np.hstack(row_imgs)
        rows.append(row)
    gallery = np.vstack(rows)
    if gallery.shape[1] > max_width:
        scale = max_width / gallery.shape[1]
        gallery = cv2.resize(gallery, (max_width, int(gallery.shape[0]*scale)))
    print('guardando en:', output_path)
    cv2.imwrite(output_path, gallery)

def make_gallery_per_person(top_faces_dir, output_dir, size=(8,8), max_width=1080):
    import cv2
    import numpy as np
    import os
    from glob import glob

    # Encuentra los IDs de las personas a partir de los nombres de los archivos
    face_imgs_all = sorted(glob(os.path.join(top_faces_dir, "tid*_face*.jpg")))
    # Ej: tid1_face0_f35.jpg -> tid1
    person_ids = set([os.path.basename(f).split('_')[0] for f in face_imgs_all])

    os.makedirs(output_dir, exist_ok=True)
    for tid in person_ids:
        face_imgs = sorted(glob(os.path.join(top_faces_dir, f"{tid}_face*.jpg")))
        gallery_imgs = [cv2.imread(f) for f in face_imgs if cv2.imread(f) is not None]
        gallery_imgs = [cv2.resize(img, (112, 112)) for img in gallery_imgs if img is not None]
        num_needed = size[0] * size[1]
        if len(gallery_imgs) < num_needed:
            black = np.zeros((112, 112, 3), dtype=np.uint8)
            gallery_imgs += [black] * (num_needed - len(gallery_imgs))
        gallery_imgs = gallery_imgs[:num_needed]
        rows = []
        for i in range(size[0]):
            row_imgs = gallery_imgs[i*size[1]:(i+1)*size[1]]
            row = np.hstack(row_imgs)
            rows.append(row)
        gallery = np.vstack(rows)
        if gallery.shape[1] > max_width:
            scale = max_width / gallery.shape[1]
            gallery = cv2.resize(gallery, (max_width, int(gallery.shape[0]*scale)))
        gallery_name = f"{tid}_gallery_8x8.jpg"
        gallery_path = os.path.join(output_dir, gallery_name)
        print('guardando galería de', tid, 'en:', gallery_path)
        cv2.imwrite(gallery_path, gallery)

def get_expanded_bbox(bbox, img_shape, scale=1.3):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_x1 = max(0, cx - new_w // 2)
    new_y1 = max(0, cy - new_h // 2)
    new_x2 = min(img_shape[1], cx + new_w // 2)
    new_y2 = min(img_shape[0], cy + new_h // 2)
    return [new_x1, new_y1, new_x2, new_y2]
