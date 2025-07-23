import os
import json
import cv2
import matplotlib.pyplot as plt
from cv.postprocess.face_indexer import FaceIndexer
from cv.postprocess.metrics_processor import MetricsProcessor
from cv.postprocess.visualization import plot_trajectory_on_frame, plot_velocity_over_time

class PostProcessor:
    def __init__(self, frames_json_path, video_name, fps=30, activity_thresh=1.0, faces_base_dir="data/top_faces"):
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

            bodies = result["person_metrics"].get("bodies", [])
            if bodies:
                matched_body_path = bodies[0]
                shutil.copy(matched_body_path, os.path.join(q_out_dir, "matched_body.jpg"))
            else:
                matched_body_path = None

            with open(os.path.join(q_out_dir, "track_id.txt"), "w") as f:
                f.write(str(result["track_id"]))
            with open(os.path.join(q_out_dir, "summary.json"), "w") as f:
                json.dump(result, f, indent=2)

            person = result["person_metrics"]
            traj_img_path = os.path.join(q_out_dir, "trajectory.png")
            vel_img_path = os.path.join(q_out_dir, "velocity.png")

            plot_trajectory_on_frame(
                video_path,
                person["centroids"],
                person["states"],
                person["frames"],
                traj_img_path
            )
            plot_velocity_over_time(
                person["velocities"],
                person["states"],
                person["frames"],
                vel_img_path
            )

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            axes[0, 0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Query")
            axes[0, 0].axis("off")

            matched_face = cv2.cvtColor(cv2.imread(result["matched_face_path"]), cv2.COLOR_BGR2RGB)
            axes[0, 1].imshow(matched_face)
            axes[0, 1].set_title("Matched Face")
            axes[0, 1].axis("off")

            if matched_body_path and os.path.exists(matched_body_path):
                matched_body = cv2.cvtColor(cv2.imread(matched_body_path), cv2.COLOR_BGR2RGB)
                axes[0, 2].imshow(matched_body)
                axes[0, 2].set_title("Matched Body")
            else:
                axes[0, 2].text(0.5, 0.5, 'No body image', horizontalalignment='center', verticalalignment='center')
            axes[0, 2].axis("off")

            traj_img = cv2.cvtColor(cv2.imread(traj_img_path), cv2.COLOR_BGR2RGB)
            axes[1, 0].imshow(traj_img)
            axes[1, 0].set_title("Trajectory")
            axes[1, 0].axis("off")

            vel_img = cv2.cvtColor(cv2.imread(vel_img_path), cv2.COLOR_BGR2RGB)
            axes[1, 1].imshow(vel_img)
            axes[1, 1].set_title("Velocity")
            axes[1, 1].axis("off")

            axes[1, 2].axis("off")

            plt.suptitle(f"Track ID: {result['track_id']} | Score: {result['score']:.4f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            final_img_path = os.path.join(q_out_dir, "final_visualization.png")
            plt.savefig(final_img_path)
            print(f"[INFO] Visualización combinada guardada en {final_img_path}")

            plt.show()

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
        img = cv2.imread(query_img_path)
        if img is None:
            print(f"[ERROR] No se pudo leer la imagen de consulta: {query_img_path}")
            raise ValueError("No se pudo cargar la imagen de consulta.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = self.indexer._get_embedding_from_crop(img_rgb)

        if embedding is None:
            raise ValueError("No se detectó rostro válido en la imagen de consulta.")

        import numpy as np
        embedding = embedding.astype("float32")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        elif embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"[ERROR] Embedding tiene forma inválida: {embedding.shape}")

        D, I = index.search(embedding, 1)

        idx = I[0][0]
        score = D[0][0]
        tid, matched_path = labels[idx]

        return {
            "track_id": tid,
            "matched_face_path": matched_path,
            "score": float(score),
            "person_metrics": self.results[str(tid)]
        }

    def get_matched_results(self, query_img_path):
        result = self.process_with_face_query(query_img_path)
        return {
            "track_id": result["track_id"],
            "matched_body_path": result["person_metrics"]["bodies"][0] if result["person_metrics"]["bodies"] else None,
            "matched_face_path": result["matched_face_path"],
            "person_metrics": result["person_metrics"]
        }
