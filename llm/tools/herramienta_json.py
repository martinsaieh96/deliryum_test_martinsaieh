from langchain.tools import Tool
from llm.agentes.agente_json import AgenteJSON

def tool_busqueda_json(entrada: str) -> str:
    """
    Espera entrada en el formato: 'track_id|ruta_json'
    """
    try:
        track_id, ruta_json = entrada.split("|")
        agente = AgenteJSON(ruta_json.strip())
        datos = agente.buscar_datos(track_id.strip())
        import json
        return json.dumps(datos, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error al procesar consulta JSON: {e}"

herramienta_busqueda_json = Tool(
    name="BusquedaDatosPersona",
    func=tool_busqueda_json,
    description="Busca los datos relevantes de una persona por track_id y ruta JSON. Entrada: 'track_id|ruta_json'. Devuelve resumen estructurado."
)
