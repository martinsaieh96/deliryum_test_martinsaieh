import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScoreEvaluator:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu'):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)

    def calcular_clipscore(self, imagen_path, descripcion):
        imagen = Image.open(imagen_path).convert("RGB")
        inputs = self.processor(text=[descripcion], images=[imagen], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.softmax(dim=1).cpu().numpy()[0][0]
        return float(score)
