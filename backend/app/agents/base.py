from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_groq import ChatGroq
from app.config import config


class A2AAgent(ABC):
    def __init__(self):
        self.llm = ChatGroq(
            model=config.PRIMARY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.7,
        )

    @abstractmethod
    def process(self, task: Dict[str, Any])-> Dict[str, Any]:
        pass    

    def _invoke_llm(self, system: str, human: str) -> str:
        response = self.llm.invoke(
            {"role": "system", "content": system},
            {"role": "user", "content": human}
        )
        return response.content