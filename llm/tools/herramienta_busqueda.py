from langchain.tools import Tool
from llm.agentes.agente_busqueda import AgenteBusqueda

buscador = AgenteBusqueda()

def tool_busqueda_ropa(descripcion: str) -> str:
    return buscador.buscar_ropa(descripcion)

herramienta_busqueda_ropa = Tool(
    name="BusquedaRopaWeb",
    func=tool_busqueda_ropa,
    description="Busca en internet ropa similar a la descripción entregada."
)
