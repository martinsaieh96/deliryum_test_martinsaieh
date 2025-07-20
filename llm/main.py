# main.py
from supervisor import Supervisor
from llm.agentes.agente_descripcion import AgenteMultimodal

multimodal_agente = AgenteMultimodal()
supervisor = Supervisor(multimodal_agente)

imagen_path = "/home/martin/deliryum_test/data/crops/1_0.jpg"

# Preguntas claras
preguntas = [
    "Describe detalladamente la vestimenta.",
    "¿Cuál es el género aparente?",
    "¿Lleva bolsas, maletas u otros objetos?"
]

resultados = {}

for pregunta in preguntas:
    tarea = f"{imagen_path} | {pregunta}"
    respuesta = supervisor.ejecutar_tarea(f"Usa AnalisisMultimodal para {tarea}")
    resultados[pregunta] = respuesta

print(resultados)
