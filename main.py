import os
import argparse
from cv.processor import VideoProcessor
from cv.postprocess.postprocessor import PostProcessor

def get_video_names(raw_video_dir):
    return [f for f in os.listdir(raw_video_dir) if f.endswith(".mp4")]

def run_task(task, video_name):
    json_path = os.path.join("data", "json", f"frames_{video_name}.json")
    config_path = "cv/config/config_processor.yaml"

    if task == "detect":
        print(f"[INFO] Ejecutando detección en {video_name}")
        vp = VideoProcessor(config_path=config_path, video_name=video_name)
        vp.run()
        return

    if not os.path.exists(json_path):
        print(f"[WARNING] No se encontró el archivo JSON para {video_name}: {json_path}")
        return

    pp = PostProcessor(frames_json_path=json_path, video_name=video_name)

    if task == "process":
        pp.process_metrics()

    elif task == "postprocess":
        query_path = input("Ruta de imagen para reconocer: ")
        if not os.path.exists(query_path):
            print("[ERROR] Imagen no encontrada.")
            return
        result = pp.process_with_face_query(query_path)
        print("\nResultado encontrado:")
        import json
        print(json.dumps(result, indent=2))

    else:
        print(f"[ERROR] Tarea desconocida: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Nombre del video a procesar (ej. cam1.mp4). Si no se especifica, se procesan todos.", default=None)
    parser.add_argument("--task", help="Tarea a realizar: process o postprocess", choices=["detect","process", "postprocess"], required=True)
    args = parser.parse_args()

    raw_video_dir = os.path.join("data", "raw_videos")

    if args.video:
        run_task(args.task, args.video)
    else:
        for video in get_video_names(raw_video_dir):
            print(f"\n[INFO] Ejecutando tarea '{args.task}' para {video}")
            run_task(args.task, video)
