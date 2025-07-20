import os
import json
from llm_analisis.agentes.agente_descripcion import AgenteDescripcion
from llm_analisis.agentes.agente_busqueda import AgenteBusqueda

class LlmAnalisis:
    def __init__(self, video_name, json_dir="data/json"):
        # video_name: "vid1.mp4" (o "vid1" si prefieres)
        self.video_name = video_name if video_name.endswith('.mp4') else video_name + '.mp4'
        self.json_path = os.path.join(json_dir, f"frames_{self.video_name}.json")
        self.descripcion_agente = AgenteDescripcion()
        self.busqueda_agente = AgenteBusqueda()
        self._json_data = None  # Lazy load

    def _load_json(self):
        if self._json_data is None:
            with open(self.json_path, 'r') as f:
                self._json_data = json.load(f)
        return self._json_data

    def _parse_ids(self, imagen_path):
        # Asume path: ./data/crops/vid1/36_0.jpg
        basename = os.path.basename(imagen_path)
        track_id = int(basename.split('_')[0])
        return track_id

    def _extraer_info_persona(self, track_id):
        data = self._load_json()
        frames_persona = [f for f in data for o in f['objects'] if o['track_id'] == track_id]
        # Puedes extraer más datos aquí: velocidad promedio, tiempo inactivo, etc.
        # Ejemplo básico: 
        n_frames = len(frames_persona)
        frame_idxs = [f['frame'] for f in data if any(o['track_id'] == track_id for o in f['objects'])]
        return {
            "track_id": track_id,
            "n_frames": n_frames,
            "frames": frame_idxs,
            # Aquí puedes poner velocidad, inactividad, etc.
        }

    def analizar_persona(self, imagen_path):
        # 1. Inferir track_id y video_name
        track_id = self._parse_ids(imagen_path)

        # 2. Descripción multimodal
        descripcion = self.descripcion_agente.analizar_vestimenta(imagen_path)
        genero = self.descripcion_agente.analizar_genero(imagen_path)
        objetos = self.descripcion_agente.analizar_objetos(imagen_path)

        # 3. Búsqueda web de ropa
        enlaces_ropa = self.busqueda_agente.buscar_ropa(descripcion)

        # 4. Consultar JSON y agregar info relevante
        datos_json = self._extraer_info_persona(track_id)

        # 5. Output estructurado
        resultado = {
            "imagen": imagen_path,
            "video": self.video_name,
            "track_id": track_id,
            "descripcion_ropa": descripcion,
            "genero_aparente": genero,
            "objetos": objetos,
            "ropa_web": enlaces_ropa,
            "datos_asociados": datos_json
        }
        return resultado
