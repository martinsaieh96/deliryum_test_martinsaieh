import os
import pandas as pd
from agentes.agente_descripcion import AgenteDescrpcion
from evaluador_clip import CLIPScoreEvaluator
from utils.logger import configurar_logger

logger = configurar_logger()

agente = AgenteDescrpcion()
clip_eval = CLIPScoreEvaluator()

# Cargar prompts
prompt_dir = "prompts"
prompts = {
    f.replace("prompt_", "").replace(".txt", ""): open(os.path.join(prompt_dir, f)).read()
    for f in os.listdir(prompt_dir) if f.startswith("prompt_")
}

# Cargar imágenes
imagenes = [f for f in os.listdir("imagenes") if f.endswith(".jpg")]

resultados = []

for imagen in imagenes:
    ruta = os.path.join("imagenes", imagen)
    for nombre_tecnica, prompt in prompts.items():
        try:
            descripcion = agente.analizar_imagen(ruta, prompt)
            clipscore = clip_eval.calcular_clipscore(ruta, descripcion)

            resultados.append({
                "imagen": imagen,
                "tecnica_prompt": nombre_tecnica,
                "descripcion_generada": descripcion,
                "clipscore": clipscore
            })

            logger.info(f"[{imagen}][{nombre_tecnica}] CLIPScore: {clipscore:.4f}")
        except Exception as e:
            logger.error(f"Error en {imagen} con {nombre_tecnica}: {e}")

# Guardar CSV
df = pd.DataFrame(resultados)
os.makedirs("resultados", exist_ok=True)
df.to_csv("resultados/resultados_clipscore.csv", index=False)
