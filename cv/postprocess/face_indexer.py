import os
import numpy as np
import cv2
import faiss
import pickle
from insightface.app import FaceAnalysis

class FaceIndexer:
    def __init__(self, faces_dir: str, video_name: str, dim: int = 512):
        # Asegura que video_name NO tenga la extensión y usa convención /faces_dir/video_name
        video_name = video_name.replace(".mp4", "")
        self.faces_dir = os.path.join(faces_dir, video_name)
        self.index_path = f"gallery_embeddings_{video_name}.faiss"
        self.label_path = f"labels_{video_name}.pkl"
        self.dim = dim
        self.face_analyzer = None  # Usaremos FaceAnalysis (no sólo el embedder)

    def _load_face_analyzer(self):
        if self.face_analyzer is None:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0)
            self.face_analyzer = app
        return self.face_analyzer

    def build_index(self):
        print(f"[INFO] Construyendo índice FAISS desde: {self.faces_dir}")
        face_paths = [os.path.join(self.faces_dir, f) for f in os.listdir(self.faces_dir) if f.lower().endswith(".jpg")]
        embeddings = []
        labels = []
        face_analyzer = self._load_face_analyzer()

        for path in face_paths:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Si no es 112x112, reescala
            if img_rgb.shape[:2] != (112, 112):
                img_rgb = cv2.resize(img_rgb, (112, 112))
            emb = self._get_embedding_from_crop(img_rgb)
            if emb is not None:
                embeddings.append(emb.astype("float32"))
                tid = int(os.path.basename(path).split("tid")[1].split("_")[0])
                labels.append((tid, path))
            else:
                print(f"[WARN] No se pudo extraer embedding de {path}")

        if not embeddings:
            raise RuntimeError("No se pudieron extraer embeddings de las imágenes.")

        emb_matrix = np.vstack(embeddings)
        index = faiss.IndexFlatIP(self.dim)
        index.add(emb_matrix)

        faiss.write_index(index, self.index_path)
        with open(self.label_path, "wb") as f:
            pickle.dump(labels, f)

        print(f"[INFO] Índice guardado: {self.index_path} ({len(labels)} rostros)")

    def _get_embedding_from_crop(self, img_rgb):
        face_analyzer = self._load_face_analyzer()
        faces = face_analyzer.get(img_rgb)
        if faces and hasattr(faces[0], "embedding"):
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)
        # Fall-back: intenta extraer embedding directo (necesita un bbox)
        try:
            h, w = img_rgb.shape[:2]
            bbox = np.array([0, 0, w, h])
            emb = face_analyzer.models['recognition'].get(img_rgb, bbox)
            if emb is not None:
                return emb / np.linalg.norm(emb)
        except Exception as e:
            print(f"[WARN] Error obteniendo embedding directo: {e}")
        return None



    def index_exists(self):
        return os.path.exists(self.index_path) and os.path.exists(self.label_path)

    def load_index(self):
        index = faiss.read_index(self.index_path)
        with open(self.label_path, "rb") as f:
            labels = pickle.load(f)
        return index, labels

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_analyzer = self._load_face_analyzer()
        faces = face_analyzer.get(img_rgb)
        if faces and hasattr(faces[0], "embedding"):
            emb = faces[0].embedding
            return emb / np.linalg.norm(emb)
        return None
