import os
import numpy as np
import cv2
import faiss
import pickle
from insightface.app import FaceAnalysis

class FaceIndexer:
    def __init__(self, faces_dir: str, video_name: str, dim: int = 512):
        self.faces_dir = os.path.join(faces_dir, video_name)
        self.index_path = f"gallery_embeddings_{video_name}.faiss"
        self.label_path = f"labels_{video_name}.pkl"
        self.dim = dim
        self.embedder = None

    def _load_embedder(self):
        if self.embedder is None:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0)
            self.embedder = app.models['recognition']
        return self.embedder

    def build_index(self):
        print(f"[INFO] Construyendo índice FAISS desde: {self.faces_dir}")
        face_paths = [os.path.join(self.faces_dir, f) for f in os.listdir(self.faces_dir) if f.endswith(".jpg")]
        embeddings = []
        labels = []

        embedder = self._load_embedder()

        for path in face_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = embedder.get(img)
            if emb is not None:
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb.astype("float32"))
                tid = int(path.split("tid")[1].split("_")[0])
                labels.append((tid, path))

        if not embeddings:
            raise RuntimeError("No se pudieron extraer embeddings de las imágenes.")

        emb_matrix = np.vstack(embeddings)
        index = faiss.IndexFlatIP(self.dim)
        index.add(emb_matrix)

        faiss.write_index(index, self.index_path)
        with open(self.label_path, "wb") as f:
            pickle.dump(labels, f)

        print(f"[INFO] Índice guardado: {self.index_path} ({len(labels)} rostros)")

    def index_exists(self):
        return os.path.exists(self.index_path) and os.path.exists(self.label_path)

    def load_index(self):
        index = faiss.read_index(self.index_path)
        with open(self.label_path, "rb") as f:
            labels = pickle.load(f)
        return index, labels

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedder = self._load_embedder()
        emb = embedder.get(img)
        if emb is not None:
            return emb / np.linalg.norm(emb)
        return None
