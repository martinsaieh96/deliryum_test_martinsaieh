from duckduckgo_search import DDGS

class AgenteBusqueda:
    def __init__(self):
        self.ddgs = DDGS()

    def buscar_ropa(self, descripcion, max_results=2):
        query = f"ropa similar a: {descripcion}"
        resultados = self.ddgs.text(query, max_results=max_results)
        enlaces = []
        for r in resultados:
            titulo = r.get("title", "")
            url = r.get("href", "")
            if titulo and url:
                enlaces.append({"titulo": titulo, "url": url})
        return enlaces if enlaces else [{"titulo": "No se encontraron enlaces relevantes.", "url": ""}]
