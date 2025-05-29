from pydantic_ai_client import PydanticAIClient
from langchain_client import LangChainClient
from langgraph_client import LangGraphClient
from ollama_manager import OllamaManager  # <-- Add this import


def main():
    prompt = "Given a DataFrame with columns 'product' and 'sales', calculate the total sales for each product."
    model_name = "qwen2.5:0.5b"

    # Ensure Ollama is running and the model is available
    # Use api_mode="openai" if you want to use PydanticAI
    manager = OllamaManager(model_name=model_name, api_mode="openai")
    if not manager.ensure_ollama_and_model():
        print("Ollama or the required model is not available. Exiting.")
        return

    clients = [
        PydanticAIClient(model_name),
        LangChainClient(model_name),
        LangGraphClient(model_name),
    ]

    for client in clients:
        print(f"\n--- {client.__class__.__name__} ---")
        print(client.run(prompt))


if __name__ == "__main__":
    main()
