class TopFacesManager:
    def __init__(self, max_faces=3, use_score=False):
        self.max_faces = max_faces
        self.use_score = use_score
        self.faces_dict = {}  # {track_id: [ (score_area, image, frame_idx), ... ]}

    def consider(self, track_id, face_img, frame_idx, score=1.0):
        h, w = face_img.shape[:2]
        area = w * h
        score_area = score * area if self.use_score else area

        if track_id not in self.faces_dict:
            self.faces_dict[track_id] = []

        current_faces = self.faces_dict[track_id]

        # Evitar duplicados por frame (opcional)
        if any(f[2] == frame_idx for f in current_faces):
            return

        current_faces.append((score_area, face_img, frame_idx))
        current_faces = sorted(current_faces, key=lambda x: x[0], reverse=True)[:self.max_faces]
        self.faces_dict[track_id] = current_faces

    def save_faces(self, output_dir):
        import os, cv2
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        for tid, faces in self.faces_dict.items():
            saved_paths[tid] = []
            for i, (score_area, img, frame_idx) in enumerate(faces):
                fname = f"tid{tid}_face{i}_f{frame_idx}.jpg"
                fpath = os.path.join(output_dir, fname)
                cv2.imwrite(fpath, img[:,:,::-1])  # RGB -> BGR para OpenCV
                saved_paths[tid].append(fpath)
        return saved_paths
