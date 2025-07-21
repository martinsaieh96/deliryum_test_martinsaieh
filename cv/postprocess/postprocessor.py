import os
import json
from cv.postprocess.face_indexer import FaceIndexer
from cv.postprocess.metrics_processor import MetricsProcessor
from cv.postprocess.visualization import plot_trajectory_on_frame, plot_velocity_over_time

class PostProcessor:
    def __init__(self, frames_json_path, video_name, fps=30, activity_thresh=1.0, faces_base_dir="data/faces"):
        self.video_name = video_name
        self.faces_dir = faces_base_dir
        self.frames_json_path = frames_json_path
        self.index_path = f"gallery_embeddings_{video_name}.faiss"
        self.label_path = f"labels_{video_name}.pkl"
        self.output_path = f"results_summary_{video_name}.json"
        self.fps = fps
        self.activity_thresh = activity_thresh
        self.results = None
        self.metrics_processor = MetricsProcessor(frames_json_path, fps=fps, activity_thresh=activity_thresh)
        self.indexer = FaceIndexer(self.faces_dir, self.video_name)

    def bulk_postprocess(self, query_dir=None, output_dir=None):
        video_name = self.video_name
        video_short = video_name.replace(".mp4", "")
        if query_dir is None:
            query_dir = os.path.join("data", "search", video_short, "queries")
        if output_dir is None:
            output_dir = os.path.join("data", "search", video_short, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = os.path.join("data", "raw_videos", video_name)

        if not self.indexer.index_exists():
            self.indexer.build_index()
        self.process_metrics()

        query_imgs = [f for f in os.listdir(query_dir) if f.lower().endswith((".jpg", ".png"))]
        for idx, img_name in enumerate(sorted(query_imgs)):
            img_path = os.path.join(query_dir, img_name)
            result = self.process_with_face_query(img_path)

            q_out_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
            os.makedirs(q_out_dir, exist_ok=True)

            import shutil
            shutil.copy(result["matched_face_path"], os.path.join(q_out_dir, "matched_face.jpg"))
            with open(os.path.join(q_out_dir, "track_id.txt"), "w") as f:
                f.write(str(result["track_id"]))
            with open(os.path.join(q_out_dir, "summary.json"), "w") as f:
                json.dump(result, f, indent=2)
            person = result["person_metrics"]
            plot_trajectory_on_frame(
                video_path,
                person["centroids"],
                person["states"],
                person["frames"],
                os.path.join(q_out_dir, "trajectory.png")
            )
            plot_velocity_over_time(
                person["velocities"],
                person["states"],
                person["frames"],
                os.path.join(q_out_dir, "velocity.png")
            )
        print(f"[INFO] Procesamiento bulk terminado: {len(query_imgs)} queries.")

    def process_metrics(self):
        self.results = self.metrics_processor.compute_metrics()
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[INFO] Métricas calculadas y guardadas en {self.output_path}")

    def process_with_face_query(self, query_img_path):
        if self.results is None:
            self.process_metrics()
            with open(self.output_path) as f:
                self.results = json.load(f)

        if not self.indexer.index_exists():
            self.indexer.build_index()

        index, labels = self.indexer.load_index()
        embedding = self.indexer.get_embedding(query_img_path)
        if embedding is None:
            raise ValueError("No se detectó rostro válido en la imagen de consulta.")

        import numpy as np
        D, I = index.search(np.array([embedding.astype("float32")]), 1)
        idx = I[0][0]
        score = D[0][0]
        tid, matched_path = labels[idx]

        return {
            "track_id": tid,
            "matched_face_path": matched_path,
            "score": float(score),
            "person_metrics": self.results[str(tid)]
        }
