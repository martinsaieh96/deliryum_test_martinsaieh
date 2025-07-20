import json
import pickle

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
