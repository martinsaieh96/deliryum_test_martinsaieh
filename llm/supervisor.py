from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from llm.tools.herramienta_descripcion import (
    herramienta_vestimenta, herramienta_genero, herramienta_objetos
)
from llm.tools.herramienta_busqueda import herramienta_busqueda_ropa
from llm.tools.herramienta_json import herramienta_busqueda_json

class Supervisor:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.tools = [
            herramienta_vestimenta,
            herramienta_genero,
            herramienta_objetos,
            herramienta_busqueda_ropa,
            herramienta_busqueda_json,
        ]
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # O 'conversational-react-description'
            verbose=True
        )

    def ejecutar(self, prompt: str) -> str:
        return self.agent.run(prompt)
