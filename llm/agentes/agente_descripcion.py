import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
class AgenteDescripcion:
    def __init__(self, modelo="Salesforce/blip-image-captioning-large", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(modelo)
        self.model = BlipForConditionalGeneration.from_pretrained(modelo).to(self.device)

    def _run(self, imagen_path, prompt):
        imagen = Image.open(imagen_path).convert("RGB")
        inputs = self.processor(text=prompt, images=imagen, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)
        respuesta = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return respuesta

    def analizar_vestimenta(self, imagen_path):
        prompt = (
            "Describe de forma clara y precisa la vestimenta visible de la persona en esta imagen. "
            "Incluye color, tipo de prenda y detalles relevantes."
        )
        return self._run(imagen_path, prompt)

    def analizar_genero(self, imagen_path):
        prompt = (
            "¿La persona en la imagen aparenta ser hombre, mujer, niño, niña o no es posible determinarlo? "
            "Responde SOLO una palabra y sin explicación adicional."
        )
        return self._run(imagen_path, prompt)

    def analizar_objetos(self, imagen_path):
        prompt = (
            "¿La persona lleva bolsas de compras, maletas u otros objetos visibles? "
            "Responde con una lista separada por comas (ejemplo: 'bolsa, mochila, maleta'), o indica 'ninguno' si no ves objetos."
        )
        return self._run(imagen_path, prompt)

    def analizar_todo(self, imagen_path):
        return {
            "descripcion_ropa": self.analizar_vestimenta(imagen_path),
            "genero_aparente": self.analizar_genero(imagen_path),
            "objetos": self.analizar_objetos(imagen_path),
        }
