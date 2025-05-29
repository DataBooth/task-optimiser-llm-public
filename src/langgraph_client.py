from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from model_client import ModelClient


class LLMState(TypedDict):
    prompt: str
    result: str


class LangGraphClient(ModelClient):
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

        def node_fn(state: LLMState):
            prompt = state["prompt"]
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]
            response = self.chat_model.invoke(messages)
            return {"prompt": prompt, "result": response.content}

        graph_builder = StateGraph(LLMState)
        graph_builder.add_node("run_llm", node_fn)
        graph_builder.add_edge("run_llm", END)
        graph_builder.set_entry_point("run_llm")
        compiled_graph = graph_builder.compile()
        self.graph = compiled_graph  # <--- store the compiled graph

    def run(self, prompt):
        result = self.graph.invoke({"prompt": prompt, "result": ""})
        return result["result"]
