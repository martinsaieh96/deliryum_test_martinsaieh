import os
import json
import argparse
from cv.processor import VideoProcessor
from cv.postprocess.postprocessor import PostProcessor
from llm.supervisor import Supervisor

def get_video_names(raw_video_dir):
    return [f for f in os.listdir(raw_video_dir) if f.endswith(".mp4")]

def run_task(task, video_name):
    json_path = os.path.join("data", "json", f"persons_{video_name}.json")
    config_path = "cv/config/config_processor.yaml"

    if task == "p1":
        print(f"[INFO] Ejecutando detección en {video_name}")
        vp = VideoProcessor(config_path=config_path, video_name=video_name)
        vp.run()
        return

    if not os.path.exists(json_path):
        print(f"[WARNING] No se encontró el archivo JSON para {video_name}: {json_path}")
        return

    elif args.task == "p2":
        print(f"[INFO] Ejecutando PostProcessor para: {args.video}")
        pp = PostProcessor(frames_json_path=json_path, video_name=args.video)
        pp.bulk_postprocess()

    elif task == "p3":
        query_dir = os.path.join("data", "search", args.video.replace(".mp4",""), "queries")
        if not os.path.exists(query_dir):
            print(f"[ERROR] No existe la carpeta de queries: {query_dir}")
            return

        query_imgs = [f for f in os.listdir(query_dir) if f.lower().endswith((".jpg", ".png"))]
        if not query_imgs:
            print("[INFO] No se encontraron imágenes de consulta.")
            return
    
        pp = PostProcessor(frames_json_path=json_path, video_name=args.video)
        supervisor = Supervisor(video_name=args.video)

        for query_img_name in sorted(query_imgs):
            print(f"\n[INFO] Procesando query: {query_img_name}")
            try:
                query_dir = os.path.join("data", "search", video_name.replace(".mp4",""), "queries")
                query_img_path = os.path.join(query_dir, query_img_name) 
                matched_results = pp.get_matched_results(query_img_path)
                track_id = matched_results["track_id"]
                matched_body_img = matched_results["matched_body_path"]

                print(f"[INFO] Track ID: {track_id}")
                print(f"[INFO] Imagen cuerpo coincidente: {matched_body_img}")

                respuesta_llm = supervisor.analizar_persona(track_id)

                output_dir = os.path.join(
                    "data", "llm_analysis", args.video.replace(".mp4", ""), f"query_{os.path.splitext(query_img_name)[0]}"
                )
                os.makedirs(output_dir, exist_ok=True)

                # Guarda el output del LLM
                with open(os.path.join(output_dir, f"llm_output_tid{track_id}.txt"), "w") as f:
                    f.write(respuesta_llm)

                # (Opcional) Copia también la imagen consulta y el cuerpo detectado para referencia
                import shutil
                shutil.copy(os.path.join(query_dir, query_img_name), os.path.join(output_dir, "query_img.jpg"))
                if matched_body_img and os.path.exists(matched_body_img):
                    shutil.copy(matched_body_img, os.path.join(output_dir, "matched_body.jpg"))

                print(f"[INFO] LLM output guardado en {output_dir}")

            except Exception as e:
                print(f"[ERROR] Falló procesamiento para {query_img_name}: {e}")

    else:
        print(f"[ERROR] Tarea desconocida: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Nombre del video a procesar (ej. cam1.mp4). Si no se especifica, se procesan todos.", default=None)
    parser.add_argument("--task", help="Tarea a realizar: process o postprocess", choices=["p1", "p2", "p3"], required=True)

    args = parser.parse_args()

    raw_video_dir = os.path.join("data", "raw_videos")

    if args.video:
        run_task(args.task, args.video)
    else:
        for video in get_video_names(raw_video_dir):
            print(f"\n[INFO] Ejecutando tarea '{args.task}' para {video}")
            run_task(args.task, video)
