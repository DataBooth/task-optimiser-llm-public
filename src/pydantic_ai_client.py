from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent
from model_client import ModelClient


class PydanticAIClient(ModelClient):
    def __init__(
        self,
        model_name,
        temperature=0.3,
        base_url="http://localhost:11434/v1",
        system_prompt=None,
    ):
        super().__init__(model_name, temperature, base_url, system_prompt)
        self.provider = OpenAIProvider(base_url=self.base_url, api_key="ollama")
        self.model = OpenAIModel(model_name=self.model_name, provider=self.provider)
        self.model_settings = ModelSettings(temperature=self.temperature)
        self.agent = Agent(self.model, system_prompt=self.system_prompt)

    def run(self, prompt):
        result = self.agent.run_sync(prompt, model_settings=self.model_settings)
        return result.output
