import subprocess
import time
import ollama
from loguru import logger


class OllamaManager:
    def __init__(self, model_name="llama2", api_mode="native"):
        """
        api_mode: "native" (default) or "openai"
        """
        self.model_name = model_name
        self.api_mode = api_mode
        logger.add("ollama_manager.log", rotation="500 MB", level="INFO")

    def is_ollama_running(self):
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("Ollama is running.")
                return True
            else:
                logger.warning(
                    f"Ollama is not running (exit code {result.returncode})."
                )
                return False
        except FileNotFoundError:
            logger.error("Ollama is not installed or not in PATH.")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Ollama is not responding in a timely manner.")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False

    def start_ollama(self):
        try:
            # Try Homebrew first (common on macOS)
            result = subprocess.run(
                ["brew", "services", "start", "ollama"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("Ollama started successfully using brew services.")
                return True
            else:
                logger.warning(
                    f"Failed to start Ollama using brew services (exit code {result.returncode}): {result.stderr}"
                )
                # Fallback to direct serve
                serve_cmd = ["ollama", "serve"]
                if self.api_mode == "openai":
                    serve_cmd.append("--api=openai")
                result = subprocess.Popen(serve_cmd)
                logger.info(f"Ollama started using '{' '.join(serve_cmd)}'.")
                return True
        except FileNotFoundError:
            logger.error("Ollama is not installed or not in PATH.")
            return False
        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            return False

    def get_available_models(self):
        try:
            models_data = ollama.list()
            logger.debug(f"DEBUG: models_data = {models_data}")
            if "models" in models_data and isinstance(models_data["models"], list):
                model_names = [
                    m.get("name") or m.get("model") or m.get("family") or str(m)
                    for m in models_data["models"]
                ]
                logger.info(f"Available Ollama models: {model_names}")
                return model_names
            else:
                logger.error(f"Unexpected response from ollama.list(): {models_data}")
                return None
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return None

    def get_model(self):
        try:
            available_models = self.get_available_models()
            if available_models and self.model_name in available_models:
                logger.info(f"Model '{self.model_name}' is already available.")
                return True
            logger.info(f"Pulling model '{self.model_name}' from Ollama...")
            ollama.pull(self.model_name)
            logger.info(f"Model '{self.model_name}' pulled successfully.")
            return True
        except Exception as e:
            logger.error(f"Error getting model '{self.model_name}': {e}")
            return False

    def ensure_ollama_and_model(self):
        if not self.is_ollama_running():
            logger.info("Ollama not running. Attempting to start Ollama...")
            if not self.start_ollama():
                logger.error("Failed to start Ollama.")
                return False
            time.sleep(10)  # Give Ollama time to start
        if not self.get_model():
            logger.error(f"Failed to get model '{self.model_name}'.")
            return False
        return True

    def query(self, prompt):
        logger.info(f"Querying Ollama model: {self.model_name}")
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            logger.debug(f"Ollama response: {response}")
            return response["response"]
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return None


def main():
    # Choose api_mode="openai" if you want to use the OpenAI-compatible endpoint
    manager = OllamaManager(model_name="qwen2.5:0.5b", api_mode="openai")

    logger.info("Checking if Ollama is running...")
    if not manager.is_ollama_running():
        logger.info("Ollama not running. Attempting to start Ollama...")
        if manager.start_ollama():
            logger.info("Ollama started successfully.")
            time.sleep(5)
        else:
            logger.error("Failed to start Ollama. Check logs for details.")
            exit(1)
    else:
        logger.info("Ollama is running.")

    print("\nListing available models:")
    models = manager.get_available_models()

    logger.info(f"\nEnsuring model '{manager.model_name}' is available...")
    if not manager.get_model():
        logger.error(f"Failed to get or pull model '{manager.model_name}'. Exiting.")
        exit(1)
    else:
        logger.info(f"Model '{manager.model_name}' is ready.")

    logger.info("\nReady to query the model!")
    prompt = (
        "Given a DataFrame with columns 'product' and 'sales', "
        "calculate the total sales for each product."
    )
    result = manager.query(prompt)
    if result:
        print("\n--- Ollama Model Response ---")
        print(result)
    else:
        logger.error("Failed to get a response from the model. Check logs for details.")

    logger.info("\nAll actions are logged in 'ollama_manager.log'.")


if __name__ == "__main__":
    main()
