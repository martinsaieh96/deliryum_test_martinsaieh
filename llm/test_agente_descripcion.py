import os
from agentes.agente_descripcion import AgenteDescripcionBLIP
from evaluador_clip import CLIPScoreEvaluator

# Inicializar agente y evaluador
agente = AgenteDescripcionBLIP()
evaluador = CLIPScoreEvaluator()

# Ruta de la imagen de prueba
ruta_imagen = "/home/martin/deliryum_test/cv/data/crops/1_0.jpg"

# Cargar prompts
prompt_dir = "prompts"
prompts = {
    f.replace("prompt_", "").replace(".txt", ""): open(os.path.join(prompt_dir, f)).read()
    for f in os.listdir(prompt_dir) if f.startswith("prompt_")
}

# Ejecutar análisis por cada técnica de prompt
print(f"\nEvaluando imagen: {ruta_imagen}\n")

for nombre_prompt, contenido_prompt in prompts.items():
    print('prompt: ', contenido_prompt)
    try:
        print(f"🔹 Técnica: {nombre_prompt}")
        descripcion = agente.analizar_imagen(ruta_imagen, contenido_prompt)
        score = evaluador.calcular_clipscore(ruta_imagen, descripcion)

        print(f"📝 Descripción: {descripcion}")
        print(f"📊 CLIPScore: {score:.4f}\n{'-'*50}\n")
    except Exception as e:
        print(f"❌ Error con {nombre_prompt}: {e}")
