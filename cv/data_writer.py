import json
import pickle
import os

class DataWriter:
    def __init__(self, output_path, mode="json"):
        self.output_path = output_path
        self.mode = mode
        self.data = []

    def add_frame(self, frame_idx, objects_list):
        # objects_list = [dict con info de cada objeto]
        self.data.append({
            "frame": frame_idx,
            "objects": objects_list
        })

    def save(self):
        if self.mode == "json":
            with open(self.output_path, "w") as f:
                json.dump(self.data, f, indent=2)
        elif self.mode == "pkl":
            with open(self.output_path, "wb") as f:
                pickle.dump(self.data, f)
        else:
            raise ValueError("Modo no soportado")

    def save_per_person(self, out_path, top_faces_dir=None, top_bodies_dir=None, max_faces=8, max_bodies=3):
        """
        Convierte la data frame-wise a person-wise y guarda el JSON por persona.
        Se puede pasar el directorio de top faces/bodies para asociar los paths.
        """
        # Build dict by track_id
        person_dict = {}
        for entry in self.data:
            frame_idx = entry["frame"]
            for obj in entry["objects"]:
                tid = obj["track_id"]
                bbox = obj["bbox"]
                person = person_dict.setdefault(tid, {
                    "bboxes": [],
                    "frames": [],
                    "velocities": [],
                    "states": [],
                    "faces": [],
                    "bodies": []
                })
                person["bboxes"].append(bbox)
                person["frames"].append(frame_idx)
                # We'll fill velocities and states later!

        # Cálculo de velocidades y estados
        import numpy as np
        umbral_px = 2.0  # Puedes ajustar este valor

        for tid, pdata in person_dict.items():
            centroids = [( (x1+x2)//2, (y1+y2)//2 ) for x1,y1,x2,y2 in pdata["bboxes"] ]
            velocities = [0.0]
            states = ["inactivo"]
            for i in range(1, len(centroids)):
                d = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i-1]))
                velocities.append(d)
                states.append("activo" if d > umbral_px else "inactivo")
            pdata["velocities"] = velocities
            pdata["states"] = states

        # Asocia los paths de caras y cuerpos top si se pasan los directorios
        if top_faces_dir and os.path.exists(top_faces_dir):
            for tid in person_dict:
                faces_paths = sorted([f for f in os.listdir(top_faces_dir) if f.startswith(f"tid{tid}_face")])
                person_dict[tid]["faces"] = [os.path.join(top_faces_dir, f) for f in faces_paths][:max_faces]
        if top_bodies_dir and os.path.exists(top_bodies_dir):
            for tid in person_dict:
                bodies_paths = sorted([f for f in os.listdir(top_bodies_dir) if f.startswith(f"tid{tid}_body")])
                person_dict[tid]["bodies"] = [os.path.join(top_bodies_dir, f) for f in bodies_paths][:max_bodies]

        with open(out_path, "w") as f:
            json.dump(person_dict, f, indent=2)
