class ModelClient:
    DEFAULT_SYSTEM_PROMPT = "You are a data scientist who writes efficient Python code with short docstring."

    def __init__(
        self,
        model_name,
        temperature=0.3,
        base_url="http://localhost:11434",
        system_prompt=None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def run(self, prompt):
        raise NotImplementedError("Subclasses must implement this method.")
