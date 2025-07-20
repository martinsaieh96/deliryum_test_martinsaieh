from langchain.tools import Tool
from llm.agentes.agente_descripcion import AgenteDescripcion

modelo = AgenteDescripcion()

def tool_vestimenta(entrada: str) -> str:
    return modelo.analizar_vestimenta(entrada.strip())

def tool_genero(entrada: str) -> str:
    return modelo.analizar_genero(entrada.strip())

def tool_objetos(entrada: str) -> str:
    return modelo.analizar_objetos(entrada.strip())

herramienta_vestimenta = Tool(
    name="DescripcionVestimenta",
    func=tool_vestimenta,
    description="Describe la vestimenta de la persona en la imagen proporcionada (ruta local a imagen)."
)
herramienta_genero = Tool(
    name="DescripcionGenero",
    func=tool_genero,
    description="Detecta el género aparente de la persona en la imagen proporcionada (ruta local a imagen)."
)
herramienta_objetos = Tool(
    name="DescripcionObjetos",
    func=tool_objetos,
    description="Detecta si la persona lleva bolsas, maletas u otros objetos en la imagen (ruta local a imagen)."
)
