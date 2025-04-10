import subprocess
import ollama
from loguru import logger


def is_ollama_running():
    """
    Checks if Ollama is running by trying to ping the server.
    Returns:
        bool: True if Ollama is running, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("Ollama is running.")
            return True
        else:
            logger.warning(f"Ollama is not running (exit code {result.returncode}).")
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


def get_available_models():
    """
    Retrieves a list of available models from Ollama.
    Returns:
        list: A list of model names, or None if an error occurs.
    """
    try:
        models_data = ollama.list()

        # Check if 'models' key exists and is a list
        if "models" in models_data and isinstance(models_data["models"], list):
            model_names = [
                model.get("name", "Unknown") for model in models_data["models"]
            ]  # Use get()
            logger.info(f"Available Ollama models: {model_names}")
            return model_names
        else:
            logger.error(f"Unexpected response from ollama.list(): {models_data}")
            return None

    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return None


def start_ollama():
    """
    Starts the Ollama service.
    Returns:
        bool: True if Ollama started successfully, False otherwise.
    """
    try:
        # This assumes ollama is in the PATH. The method to start ollama may differ
        # based on how it was installed (e.g., brew services start ollama).
        subprocess.run(["ollama", "serve"], detach=True)  # Run in background

        # Wait for a short period to allow Ollama to start (adjust as needed).
        import time

        time.sleep(5)

        if is_ollama_running():
            logger.info("Ollama started successfully.")
            return True
        else:
            logger.error("Ollama failed to start.")
            return False

    except FileNotFoundError:
        logger.error("Ollama is not installed or not in PATH.")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama: {e}")
        return False


def get_model(model_name):
    """
    Pulls a model from Ollama, if it's not already available
    Args:
        model_name (str): The name of the model to pull.
    Returns:
        bool: True if the model is available (either already present or successfully pulled), False otherwise.
    """
    try:
        available_models = get_available_models()
        if available_models and model_name in available_models:
            logger.info(f"Model '{model_name}' is already available.")
            return True

        logger.info(f"Pulling model '{model_name}' from Ollama...")
        ollama.pull(model_name)  # Pull the model
        logger.info(f"Model '{model_name}' pulled successfully.")
        return True

    except Exception as e:
        logger.error(f"Error getting model '{model_name}': {e}")
        return False


def ensure_ollama_and_model(model_name="llama2"):
    """
    Ensures that Ollama is running and the specified model is available.
    Starts Ollama if it's not running and pulls the model if it's not available.
    Args:
        model_name (str): The name of the model to ensure.
    Returns:
        bool: True if Ollama is running and the model is available, False otherwise.
    """
    if not is_ollama_running():
        if not start_ollama():
            logger.error("Failed to start Ollama.")
            return False

    if not get_model(model_name):
        logger.error(f"Failed to get model '{model_name}'.")
        return False

    return True  # Ollama is running and the model is available


# Example Usage:
if __name__ == "__main__":
    logger.add(
        "ollama_manager.log", rotation="500 MB", level="INFO"
    )  # Add logging to a file
    model_name = "llama2"  # set model here
    if ensure_ollama_and_model(model_name):
        logger.info(f"Ollama is running and model '{model_name}' is available.")
        # Your code that uses Ollama and the model goes here
        # You can now proceed with your scheduling tasks
    else:
        logger.error("Ollama or the required model is not available. Exiting.")
