import json
from loguru import logger
from ollama_manager import OllamaManager

from task_module import (
    load_tasks_from_csv,
    load_calendar_entries_from_csv,
    TaskAppUserPreferences,
    Calendar,
)

OLLAMA_MODEL_NAME = "llama2:latest"


def main():
    # 1. Load data
    tasks = load_tasks_from_csv()
    calendar_entries = load_calendar_entries_from_csv()
    preferences = TaskAppUserPreferences(max_daily_hours=8, timezone="Australia/Sydney")

    # 2. Build Calendar object
    calendar = Calendar(id="1", name="Work Calendar", appointments=calendar_entries)

    # 3. Initialize OllamaManager
    ollama_manager = OllamaManager(model_name=OLLAMA_MODEL_NAME)
    if not ollama_manager.ensure_ollama_and_model():
        logger.error("Ollama or the required model is not available. Exiting.")
        exit(1)

    # 4. Prepare prompt for the LLM
    scheduler_input = {
        "tasks": [task.to_dict() for task in tasks],
        "calendar": calendar.to_dict(),
        "preferences": preferences.to_dict(),
    }
    prompt = (
        "Given the following tasks, calendar, and user preferences, "
        "provide a summary or optimized schedule:\n"
        f"{json.dumps(scheduler_input, indent=2)}"
    )

    # 5. Query Ollama
    response = ollama_manager.query(prompt)
    print("\n--- Ollama Model Response ---")
    print(response)

    logger.info("All actions are logged in 'ollama_manager.log'.")


if __name__ == "__main__":
    main()
