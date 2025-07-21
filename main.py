import os
import json
import argparse
from cv.processor import VideoProcessor
from cv.postprocess.postprocessor import PostProcessor
from llm.supervisor import Supervisor

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

    elif args.task == "postprocess":
        pp = PostProcessor(frames_json_path=json_path, video_name=args.video)
        pp.bulk_postprocess()

    elif task == "supervisor_llm":
        query_path = input("Ruta de imagen de consulta para LLM Supervisor: ")
        if not os.path.exists(query_path):
            print("[ERROR] Imagen no encontrada.")
            return

        prompt = (
            f"Resumen de la persona track_id={tid}:\n"
            f"{json.dumps(datos_persona, indent=2)}\n"
            f"Rostro encontrado en: {result['matched_face_path']}\n"
            f"Describe o analiza la persona usando toda esta información."
        )

        supervisor = Supervisor()
        respuesta = supervisor.ejecutar(prompt)
        print("\n[LLM Supervisor Output]:\n")
        print(respuesta)
        with open(f"data/json/llm_output_{video_name}_tid{tid}.txt", "w") as f:
            f.write(respuesta)

    else:
        print(f"[ERROR] Tarea desconocida: {task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Nombre del video a procesar (ej. cam1.mp4). Si no se especifica, se procesan todos.", default=None)
    parser.add_argument("--task", help="Tarea a realizar: process o postprocess", choices=["detect", "postprocess", "supervisor_llm"], required=True)

    args = parser.parse_args()

    raw_video_dir = os.path.join("data", "raw_videos")

    if args.video:
        run_task(args.task, args.video)
    else:
        for video in get_video_names(raw_video_dir):
            print(f"\n[INFO] Ejecutando tarea '{args.task}' para {video}")
            run_task(args.task, video)
