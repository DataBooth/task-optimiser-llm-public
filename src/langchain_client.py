from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from model_client import ModelClient


class LangChainClient(ModelClient):
    def __init__(
        self,
        model_name,
        temperature=0.3,
        base_url="http://localhost:11434",
        system_prompt=None,
    ):
        super().__init__(model_name, temperature, base_url, system_prompt)
        self.chat_model = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    def run(self, prompt):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.chat_model.invoke(messages)
        return response.content
