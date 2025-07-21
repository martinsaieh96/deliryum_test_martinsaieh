import os
import json
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatHuggingFace
from llm.tools.herramienta_descripcion import (
    herramienta_vestimenta, herramienta_genero, herramienta_objetos
)
from llm.tools.herramienta_busqueda import herramienta_busqueda_ropa
from llm.tools.herramienta_json import herramienta_busqueda_json, AgenteJSON
from cv.postprocess.visualization import plot_trajectory_on_frame, plot_velocity_over_time

class Supervisor:
    def __init__(self, video_name):
        self.video_name = video_name.replace(".mp4", "")
        self.json_path = os.path.join("data", "json", f"persons_{video_name}")
        self.llm = ChatHuggingFace(
                    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  
                    temperature=0.2,
                    max_new_tokens=1024
                )
        self.tools = [
            herramienta_vestimenta,
            herramienta_genero,
            herramienta_objetos,
            herramienta_busqueda_ropa,
            herramienta_busqueda_json,
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def analizar_persona(self, track_id: int, show_img=True, show_graphs=True):
        agente = AgenteJSON(self.json_path)
        datos_persona = agente.buscar_datos(track_id)

        top_faces_dir = os.path.join("data", "top_faces", self.video_name)
        representativas = [f for f in os.listdir(top_faces_dir) if f"tid{track_id}_face0" in f and f.endswith(".jpg")]
        rep_img_path = os.path.join(top_faces_dir, representativas[0]) if representativas else None

        output_viz_dir = os.path.join("data", "json", "visualizations", self.video_name)
        os.makedirs(output_viz_dir, exist_ok=True)
        traj_path = os.path.join(output_viz_dir, f"traj_tid{track_id}.jpg")
        vel_path = os.path.join(output_viz_dir, f"velocity_tid{track_id}.jpg")

        centroids = datos_persona.get("centroids", [])
        states = datos_persona.get("states", ["activo"] * len(centroids))
        frames = datos_persona.get("frames_presentes", [])
        velocities = datos_persona.get("velocities", [0] * len(frames))  # si existe

        video_path = os.path.join("data", "raw_videos", f"{self.video_name}.mp4")
        if centroids and frames:
            plot_trajectory_on_frame(video_path, centroids, states, frames, traj_path)
            if velocities:
                plot_velocity_over_time(velocities, states, frames, vel_path)
            else:
                vel_path = None
        else:
            traj_path = None
            vel_path = None

        if show_img and rep_img_path and os.path.exists(rep_img_path):
            self._show_image(rep_img_path, f"Track {track_id}: Cara más representativa")
        if show_graphs and traj_path and os.path.exists(traj_path):
            self._show_image(traj_path, "Trayectoria en primer frame")
        if show_graphs and vel_path and os.path.exists(vel_path):
            self._show_image(vel_path, "Velocidad por frame")

        prompt = f"""
                    Eres un asistente experto en análisis visual y semántico de personas detectadas en video para un sistema de seguridad inteligente.

                    A continuación te entrego la información extraída automáticamente de una persona con track_id={track_id}:

                    [RESUMEN ESTRUCTURADO DEL JSON]
                    {json.dumps(datos_persona, indent=2)}

                    [IMAGEN DE ROSTRO MÁS REPRESENTATIVA]
                    Ruta local: {rep_img_path or '[NO DISPONIBLE]'}

                    [IMAGEN DE SEGUIMIENTO/TRAYECTORIA]
                    Ruta local: {traj_path or '[NO DISPONIBLE]'}

                    [GRÁFICO DE VELOCIDAD]
                    Ruta local: {vel_path or '[NO DISPONIBLE]'}

                    **Por favor realiza un análisis integral siguiendo exactamente estos pasos:**

                    1. **Describe detalladamente la vestimenta de la persona** (por colores, tipos de prenda, accesorios, calzado si es posible).
                    2. **Identifica si la persona es hombre, mujer o niño/a** y justifica tu respuesta en base a la información visual disponible.
                    3. **Menciona si lleva bolsas de compras, maletas u otros objetos** detectables.
                    4. **Resume la información relevante del JSON:**  
                    - Tiempo total de inactividad  
                    - Velocidad promedio  
                    - Frames donde aparece  
                    - Cualquier otra estadística relevante.
                    5. **Realiza una búsqueda en internet sobre la vestimenta descrita** y entrega:  
                    - Un resumen de la información encontrada sobre ese tipo de vestimenta.  
                    - **Al menos 2 enlaces de referencia** confiables.
                    6. **Finaliza con un resumen ejecutivo en máximo 3 líneas**, útil para incluirlo en un informe de seguridad.

                    **Entrega la respuesta de manera ordenada y clara, usando títulos o formato markdown si es posible para cada sección.**
                    """

        return self.agent.run(prompt)

    def _show_image(self, img_path, title="Imagen"):
        import cv2
        import matplotlib.pyplot as plt
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(title)
            plt.axis('off')
            plt.show()
        else:
            print(f"[ERROR] No se pudo cargar la imagen: {img_path}")
