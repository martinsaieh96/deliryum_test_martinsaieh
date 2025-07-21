class TopFacesManager:
    def __init__(self, max_faces=8, max_bodies=3, use_score=False):
        self.max_faces = max_faces
        self.max_bodies = max_bodies
        self.use_score = use_score
        self.faces_dict = {}

    def consider(self, track_id, face_img, frame_idx, score=1.0, bbox_cuerpo=None):
        h, w = face_img.shape[:2]
        area = w * h
        score_area = score * area if self.use_score else area

        if track_id not in self.faces_dict:
            self.faces_dict[track_id] = []

        current_faces = self.faces_dict[track_id]
        if any(f[2] == frame_idx for f in current_faces):
            return

        current_faces.append((score_area, face_img, frame_idx, bbox_cuerpo))
        current_faces = sorted(current_faces, key=lambda x: x[0], reverse=True)[:self.max_faces]
        self.faces_dict[track_id] = current_faces

    def save_faces(self, output_dir_faces, output_dir_bodies, video_path):
        import os, cv2
        os.makedirs(output_dir_faces, exist_ok=True)
        os.makedirs(output_dir_bodies, exist_ok=True)
        saved_paths = {}
        cap = cv2.VideoCapture(video_path)
        frames_cache = {}
        for tid, faces in self.faces_dict.items():
            saved_paths[tid] = {"faces": [], "bodies": []}
            for i, (score_area, img, frame_idx, bbox_cuerpo) in enumerate(faces[:self.max_faces]):
                fname = f"tid{tid}_face{i}_f{frame_idx}.jpg"
                fpath = os.path.join(output_dir_faces, fname)
                cv2.imwrite(fpath, img[:,:,::-1])
                saved_paths[tid]["faces"].append(fpath)
            for i, (score_area, img, frame_idx, bbox_cuerpo) in enumerate(faces[:self.max_bodies]):
                if bbox_cuerpo is not None:
                    if frame_idx not in frames_cache:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            print(f"[ERROR] No se pudo leer el frame {frame_idx} del video para track {tid}.")
                            frames_cache[frame_idx] = None
                        else:
                            frames_cache[frame_idx] = frame
                    frame = frames_cache[frame_idx]
                    if frame is not None:
                        x1, y1, x2, y2 = bbox_cuerpo
                        h, w = frame.shape[:2]
                        x1, x2 = max(0, x1), min(w, x2)
                        y1, y2 = max(0, y1), min(h, y2)
                        if x2 <= x1 or y2 <= y1:
                            print(f"[WARN] BBOX inválido para cuerpo (track {tid}, frame {frame_idx}): {bbox_cuerpo}")
                            continue
                        body_crop = frame[y1:y2, x1:x2]
                        if body_crop is None or body_crop.size == 0:
                            print(f"[WARN] Crop de cuerpo vacío para track {tid}, frame {frame_idx}: {bbox_cuerpo}")
                            continue
                        body_fname = f"tid{tid}_body{i}_f{frame_idx}.jpg"
                        body_fpath = os.path.join(output_dir_bodies, body_fname)
                        cv2.imwrite(body_fpath, body_crop)
                        saved_paths[tid]["bodies"].append(body_fpath)
        cap.release()
        return saved_paths
