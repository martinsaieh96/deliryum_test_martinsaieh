import json

class AgenteJSON:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()

    def _load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def buscar_datos(self, track_id, resumen_simple=True):
        track_id = str(track_id)  
        if track_id not in self.data:
            return {"error": f"No se encontró el track_id {track_id} en el JSON"}

        persona = self.data[track_id]
        frames = persona.get("frames", [])
        bboxes = persona.get("bboxes", [])
        velocities = persona.get("velocities", [])
        states = persona.get("states", [])
        crops = persona.get("faces", [])
        bodies = persona.get("bodies", [])
        centroids = [self._bbox_to_centroid(b) for b in bboxes if b]

        resumen = {
            "track_id": track_id,
            "frames_presentes": frames,
            "num_frames": len(frames),
            "frame_inicio": min(frames) if frames else None,
            "frame_fin": max(frames) if frames else None,
            "bboxes": bboxes,
            "crops": crops,
            "bodies": bodies,
            "centroids": centroids,
            "velocities": velocities,
            "states": states,
        }
        if resumen_simple:
            return resumen
        else:
            return {
                "resumen": resumen,
                "detalles": persona
            }


    def _bbox_to_centroid(self, bbox):
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return [cx, cy]
        return None
