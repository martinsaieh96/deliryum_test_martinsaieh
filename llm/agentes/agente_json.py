import json

class AgenteJSON:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()

    def _load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def buscar_datos(self, track_id, resumen_simple=True):
        track_id = int(track_id)
        frames_info = []
        for frame_info in self.data:
            frame = frame_info.get("frame")
            for obj in frame_info.get("objects", []):
                if int(obj.get("track_id", -1)) == track_id:
                    item = {
                        "frame": frame,
                        "confidence": obj.get("confidence"),
                        "bbox": obj.get("bbox"),
                        "crop_path": obj.get("crop_path"),
                        "centroid": self._bbox_to_centroid(obj.get("bbox")),
                    }
                    frames_info.append(item)
        if not frames_info:
            return {"error": f"No se encontró el track_id {track_id} en el JSON"}

        frames_presentes = [fi["frame"] for fi in frames_info]
        confs = [fi["confidence"] for fi in frames_info if fi["confidence"] is not None]
        bboxes = [fi["bbox"] for fi in frames_info]
        crops = [fi["crop_path"] for fi in frames_info]
        centroids = [fi["centroid"] for fi in frames_info if fi["centroid"] is not None]

        resumen = {
            "track_id": track_id,
            "frames_presentes": frames_presentes,
            "num_frames": len(frames_presentes),
            "frame_inicio": min(frames_presentes),
            "frame_fin": max(frames_presentes),
            "confianza_promedio": round(sum(confs) / len(confs), 3) if confs else None,
            "bboxes": bboxes,
            "crops": crops,
            "centroids": centroids,
        }
        if resumen_simple:
            return resumen
        else:
            return {
                "resumen": resumen,
                "detalles": frames_info
            }

    def _bbox_to_centroid(self, bbox):
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return [cx, cy]
        return None
