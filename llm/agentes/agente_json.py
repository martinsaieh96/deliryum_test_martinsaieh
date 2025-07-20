import json

class AgenteJSON:
    """
    Consulta y resume información de una persona (track_id) a partir del archivo JSON de detecciones por frame.
    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()

    def _load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def buscar_datos(self, track_id):
        """
        track_id: int o str (se castea)
        Devuelve un resumen con:
            - frames donde aparece
            - confianza promedio
            - primeras y últimas apariciones
            - lista de bboxes, etc.
        """
        track_id = int(track_id)
        frames_info = []
        for frame_info in self.data:
            frame = frame_info["frame"]
            for obj in frame_info["objects"]:
                if int(obj["track_id"]) == track_id:
                    frames_info.append({
                        "frame": frame,
                        "confidence": obj.get("confidence", None),
                        "bbox": obj.get("bbox", None),
                        "crop_path": obj.get("crop_path", None)
                    })
        if not frames_info:
            return {"error": f"No se encontró el track_id {track_id} en el JSON"}

        # Estadísticas útiles
        frames_presentes = [fi["frame"] for fi in frames_info]
        confs = [fi["confidence"] for fi in frames_info if fi["confidence"] is not None]
        bboxes = [fi["bbox"] for fi in frames_info]
        crops = [fi["crop_path"] for fi in frames_info]

        resumen = {
            "track_id": track_id,
            "frames_presentes": frames_presentes,
            "num_frames": len(frames_presentes),
            "frame_inicio": min(frames_presentes),
            "frame_fin": max(frames_presentes),
            "confianza_promedio": round(sum(confs) / len(confs), 3) if confs else None,
            "bboxes": bboxes,
            "crops": crops,
        }
        return resumen
