import csv
import json
import os
import subprocess
from datetime import datetime

import ollama
from loguru import logger

try:
    import ollama
except ImportError:
    print("Ollama is not installed. Please install it: pip install ollama")
    exit()


class Task:
    def __init__(
        self,
        id,
        name,
        due_date,
        duration,
        priority,
        ordinal,
        project,
        split_allowed,
        fixed,
        min_split_time,
        total_work_minutes,
        constraint,
        constraint_date,
        dtstart_offset,
    ):
        self.id = id
        self.name = name
        self.due_date = due_date
        self.duration = float(duration)  # Duration in hours
        self.priority = int(priority)
        self.ordinal = int(ordinal)
        self.project = project
        self.split_allowed = split_allowed.lower() == "true"
        self.fixed = fixed.lower() == "true"
        self.min_split_time = int(min_split_time) if min_split_time != "NULL" else None
        self.total_work_minutes = (
            int(total_work_minutes) if total_work_minutes != "NULL" else None
        )
        self.constraint = constraint
        self.constraint_date = constraint_date
        self.dtstart_offset = int(dtstart_offset)

    def __repr__(self):
        return f"Task(id={self.id}, name={self.name}, due_date={self.due_date}, duration={self.duration}, priority={self.priority}, ordinal={self.ordinal}, project={self.project}, split_allowed={self.split_allowed}, fixed={self.fixed}, min_split_time={self.min_split_time}, total_work_minutes={self.total_work_minutes}, constraint={self.constraint}, constraint_date={self.constraint_date}, dtstart_offset={self.dtstart_offset})"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class ScheduledTask(Task):
    def __init__(
        self,
        id,
        name,
        due_date,
        duration,
        priority,
        ordinal,
        project,
        split_allowed,
        fixed,
        min_split_time,
        total_work_minutes,
        constraint,
        constraint_date,
        dtstart_offset,
        scheduled_start=None,
        scheduled_end=None,
        work_code=None,
        cycle_time_events=None,
        missed_constraints=None,
        scheduled_duration_minutes=None,
    ):
        super().__init__(
            id=id,
            name=name,
            due_date=due_date,
            duration=duration,
            priority=priority,
            ordinal=ordinal,
            project=project,
            split_allowed=split_allowed,
            fixed=fixed,
            min_split_time=min_split_time,
            total_work_minutes=total_work_minutes,
            constraint=constraint,
            constraint_date=constraint_date,
            dtstart_offset=dtstart_offset,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            work_code=work_code,
            cycle_time_events=cycle_time_events,
            missed_constraints=missed_constraints,
            scheduled_duration_minutes=scheduled_duration_minutes,
        )
        self.scheduled_start = scheduled_start
        self.scheduled_end = scheduled_end
        self.work_code = work_code
        self.cycle_time_events = cycle_time_events
        self.missed_constraints = missed_constraints
        self.scheduled_duration_minutes = scheduled_duration_minutes

    def __repr__(self):
        return f"ScheduledTask(id={self.id}, name={self.name}, due_date={self.due_date}, duration={self.duration}, priority={self.priority}, ordinal={self.ordinal}, project={self.project}, split_allowed={self.split_allowed}, fixed={self.fixed}, min_split_time={self.min_split_time}, total_work_minutes={self.total_work_minutes}, constraint={self.constraint}, constraint_date={self.constraint_date}, dtstart_offset={self.dtstart_offset}, scheduled_start={self.scheduled_start}, scheduled_end={self.scheduled_end}, work_code={self.work_code}, cycle_time_events={self.cycle_time_events}, missed_constraints={self.missed_constraints}, scheduled_duration_minutes={self.scheduled_duration_minutes})"

    def to_dict(self):
        task_dict = super().to_dict()
        task_dict.update(
            {
                "scheduled_start": self.scheduled_start,
                "scheduled_end": self.scheduled_end,
                "work_code": self.work_code,
                "cycle_time_events": self.cycle_time_events,
                "missed_constraints": self.missed_constraints,
                "scheduled_duration_minutes": self.scheduled_duration_minutes,
            }
        )
        return task_dict

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class Appointment:
    def __init__(self, id, event_name, start_time, end_time):
        self.id = id
        self.event_name = event_name
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"Appointment(id={self.id}, event_name={self.event_name}, start_time={self.start_time}, end_time={self.end_time})"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class RecurringAppointment(Appointment):
    def __init__(
        self, start_time, end_time, title=None, description=None, recurrence=None
    ):
        super().__init__(start_time, end_time, title, description)
        self.recurrence = recurrence

    def __repr__(self):
        return f"RecurringAppointment(start_time={self.start_time}, end_time={self.end_time}, title={self.title}, description={self.description}, recurrence={self.recurrence})"

    def to_dict(self):
        app_dict = super().to_dict()
        app_dict.update({"recurrence": self.recurrence})
        return app_dict

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class Project:
    def __init__(
        self,
        id,
        name,
        description=None,
        daily_max_hours=None,
        weekly_max_hours=None,
        priority=None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.daily_max_hours = daily_max_hours
        self.weekly_max_hours = weekly_max_hours
        self.priority = priority

    def __repr__(self):
        return f"Project(id={self.id}, name={self.name}, description={self.description}, daily_max_hours={self.daily_max_hours}, weekly_max_hours={self.weekly_max_hours}, priority={self.priority})"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class WorkingHours:
    def __init__(self, day, start_time, end_time):
        self.day = day
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        return f"WorkingHours(day={self.day}, start_time={self.start_time}, end_time={self.end_time})"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class Calendar:
    def __init__(self, id, name, appointments=None):
        self.id = id
        self.name = name
        self.appointments = appointments if appointments is not None else []

    def __repr__(self):
        return f"Calendar(id={self.id}, name={self.name}, appointments={self.appointments})"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "appointments": [app.to_dict() for app in self.appointments],
        }

    @classmethod
    def from_dict(cls, data):
        appointments_data = data.get("appointments", [])
        appointments = [
            Appointment.from_dict(app_data)
            if "recurrence" not in app_data
            else RecurringAppointment.from_dict(app_data)
            for app_data in appointments_data
        ]

        return cls(id=data["id"], name=data["name"], appointments=appointments)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class TaskSchedulerInput:
    def __init__(self, tasks, calendar, preferences):
        self.tasks = tasks
        self.calendar = calendar
        self.preferences = preferences

    def __repr__(self):
        return f"TaskSchedulerInput(tasks={self.tasks}, calendar={self.calendar}, preferences={self.preferences})"

    def to_dict(self):
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "calendar": self.calendar.to_dict(),
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data):
        tasks_data = data.get("tasks", [])
        tasks = [Task.from_dict(task_data) for task_data in tasks]
        calendar_data = data.get("calendar", {})
        calendar = Calendar.from_dict(calendar_data)
        preferences = data.get("preferences", {})

        return cls(tasks=tasks, calendar=calendar, preferences=preferences)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class Tasks:
    def __init__(self, tasks):
        self.tasks = tasks

    def __repr__(self):
        return f"Tasks(tasks={self.tasks})"

    def to_dict(self):
        return {"tasks": [task.to_dict() for task in self.tasks]}

    @classmethod
    def from_dict(cls, data):
        tasks_data = data.get("tasks", [])
        tasks = [Task.from_dict(task_data) for task_data in tasks]
        return cls(tasks=tasks)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


class TaskAppUserPreferences:
    def __init__(self, max_daily_hours, timezone):
        self.max_daily_hours = max_daily_hours
        self.timezone = timezone

    def __repr__(self):
        return f"TaskAppUserPreferences(max_daily_hours={self.max_daily_hours}, timezone={self.timezone})"

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


def load_tasks_from_csv(filename="tasks.csv", data_dir="data"):
    tasks = []
    filepath = os.path.join(data_dir, filename)
    try:
        with open(filepath, mode="r") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                tasks.append(Task(**row))
        logger.info(f"Loaded {len(tasks)} tasks from {filepath}")
    except Exception as e:
        logger.error(f"Error loading tasks from CSV: {e}")
        return None
    return tasks


def load_calendar_entries_from_csv(filename="calendar_entries.csv", data_dir="data"):
    calendar_entries = []
    filepath = os.path.join(data_dir, filename)
    try:
        with open(filepath, mode="r") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                calendar_entries.append(
                    Appointment(**row)
                )  # Use Appointment class here
        logger.info(f"Loaded {len(calendar_entries)} calendar entries from {filepath}")
    except Exception as e:
        logger.error(f"Error loading calendar entries from CSV: {e}")
        return None
    return calendar_entries


def load_preferences_from_csv(filename="preferences.csv", data_dir="data"):
    filepath = os.path.join(data_dir, filename)
    try:
        with open(filepath, mode="r") as file:
            csv_file = csv.DictReader(file)
            for row in csv_file:
                logger.info(f"Loaded preferences from {filepath}")
                return TaskAppUserPreferences(
                    **row
                )  # Return TaskAppUserPreferences object
    except Exception as e:
        logger.error(f"Error loading preferences from CSV: {e}")
        return None
    return None


def serialize_tasks(tasks):
    return [task.to_dict() for task in tasks]


def serialize_calendar_entries(entries):
    return [entry.to_dict() for entry in entries]


def create_prompt(task_scheduler_input):
    tasks = task_scheduler_input.tasks
    calendar = task_scheduler_input.calendar
    preferences = task_scheduler_input.preferences

    tasks_detail = "\n".join(
        [
            f"- Task ID: {task.id}, Name: {task.name}, Due Date: {task.due_date}, Duration: {task.duration} hours, Priority: {task.priority}"
            for task in tasks
        ]
    )

    calendar_entries_json = json.dumps(
        [entry.to_dict() for entry in calendar.appointments], indent=2
    )
    preferences_json = json.dumps(preferences.to_dict(), indent=2)

    return f"""
You are a scheduling assistant tasked with optimizing the sequence of tasks while respecting constraints.

Here are the tasks:
{tasks_detail}

Here are the existing calendar entries:
{calendar_entries_json}

Here are the user preferences:
{preferences_json}

Constraints:
- Tasks must not overlap with existing calendar entries.
- Tasks should be scheduled in order of priority (lower numbers are higher priority).
- Max daily working hours: {preferences.max_daily_hours} hours.
- Tasks with earlier due dates should be prioritized if possible.

Please provide an optimized schedule as a JSON object with the following format:
[
  {{
    "task_id": "Task ID",
    "scheduled_start": "YYYY-MM-DDTHH:MM",
    "scheduled_end": "YYYY-MM-DDTHH:MM"
  }}
]
"""


def query_ollama(prompt, model="llama2"):
    logger.info(f"Querying Ollama model: {model}")
    try:
        response = ollama.generate(model=model, prompt=prompt)
        logger.debug(f"Ollama response: {response}")
        return response[
            "response"
        ]  # Assuming Ollama returns a dictionary with 'response' key.
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return None


def is_ollama_model_available(model_name):
    """
    Checks if a specific model is available in Ollama.
    Args:
        model_name (str): The name of the model to check.
    Returns:
        bool: True if the model is available, False otherwise.
    """
    try:
        # Fetch the list of available models
        models_data = ollama.list()

        # Debugging: Log the raw response to understand its structure
        logger.debug(f"Ollama list response: {models_data}")

        # Check if 'models' key exists and is a list
        if "models" in models_data and isinstance(models_data["models"], list):
            # Extract model names safely
            model_names = [
                model.model if hasattr(model, "model") else "Unknown"
                for model in models_data["models"]
            ]  # HERE
            logger.info(f"Available Ollama models: {model_names}")
            return model_name in model_names
        else:
            # Log unexpected structure
            logger.error(
                f"Unexpected response format from ollama.list(): {models_data}"
            )
            return False

    except Exception as e:
        logger.error(f"Error checking Ollama model availability: {e}")
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
        # Use brew services if installed via Homebrew, otherwise use ollama serve
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

            # If brew services fails, try starting with 'ollama serve' directly
            result = subprocess.run(["ollama", "serve"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Ollama started successfully using 'ollama serve'.")
                return True
            else:
                logger.error(
                    f"Failed to start Ollama using 'ollama serve' (exit code {result.returncode}): {result.stderr}"
                )
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


class RescheduledTask(ScheduledTask):
    def __init__(
        self,
        task_id,
        scheduled_start,
        scheduled_end,
        id,
        name,
        due_date,
        duration,
        priority,
        ordinal,
        project,
        split_allowed,
        fixed,
        min_split_time,
        total_work_minutes,
        constraint,
        constraint_date,
        dtstart_offset,
        work_code=None,
        cycle_time_events=None,
        missed_constraints=None,
        scheduled_duration_minutes=None,
    ):
        super().__init__(
            id=id,
            name=name,
            due_date=due_date,
            duration=duration,
            priority=priority,
            ordinal=ordinal,
            project=project,
            split_allowed=split_allowed,
            fixed=fixed,
            min_split_time=min_split_time,
            total_work_minutes=total_work_minutes,
            constraint=constraint,
            constraint_date=constraint_date,
            dtstart_offset=dtstart_offset,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            work_code=work_code,
            cycle_time_events=cycle_time_events,
            missed_constraints=missed_constraints,
            scheduled_duration_minutes=scheduled_duration_minutes,
        )
        self.task_id = task_id
        self.scheduled_start = scheduled_start
        self.scheduled_end = scheduled_end

    def __repr__(self):
        return f"RescheduledTask(task_id={self.task_id}, scheduled_start={self.scheduled_start}, scheduled_end={self.scheduled_end})"

    def to_dict(self):
        rescheduled_dict = super().to_dict()
        rescheduled_dict.update(
            {
                "task_id": self.task_id,
                "scheduled_start": self.scheduled_start,
                "scheduled_end": self.scheduled_end,
            }
        )
        return rescheduled_dict

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)


def parse_schedule(response):
    try:
        schedule_data = json.loads(response)
        rescheduled_tasks = []
        for entry in schedule_data:
            # Extract fields for RescheduledTask, use .get() with defaults to handle missing data
            task_id = entry.get("task_id")
            scheduled_start = entry.get("scheduled_start")
            scheduled_end = entry.get("scheduled_end")

            # The Task information, use .get() with defaults
            id = entry.get("id", "default_id")
            name = entry.get("name", "default_name")
            due_date = entry.get("due_date", "default_due_date")
            duration = entry.get("duration", 0.0)
            priority = entry.get("priority", 5)
            ordinal = entry.get("ordinal", 1)
            project = entry.get("project", "default_project")
            split_allowed = entry.get("split_allowed", "false")
            fixed = entry.get("fixed", "false")
            min_split_time = entry.get("min_split_time", None)
            total_work_minutes = entry.get("total_work_minutes", None)
            constraint = entry.get("constraint", None)
            constraint_date = entry.get("constraint_date", None)
            dtstart_offset = entry.get("dtstart_offset", 0)

            # Add additional fields for ScheduledTask, if available
            work_code = entry.get("work_code", None)
            cycle_time_events = entry.get("cycle_time_events", None)
            missed_constraints = entry.get("missed_constraints", None)
            scheduled_duration_minutes = entry.get("scheduled_duration_minutes", None)

            # Construct the RescheduledTask
            rescheduled_task = RescheduledTask(
                task_id=task_id,
                scheduled_start=scheduled_start,
                scheduled_end=scheduled_end,
                id=id,
                name=name,
                due_date=due_date,
                duration=duration,
                priority=priority,
                ordinal=ordinal,
                project=project,
                split_allowed=split_allowed,
                fixed=fixed,
                min_split_time=min_split_time,
                total_work_minutes=total_work_minutes,
                constraint=constraint,
                constraint_date=constraint_date,
                dtstart_offset=dtstart_offset,
                work_code=work_code,
                cycle_time_events=cycle_time_events,
                missed_constraints=missed_constraints,
                scheduled_duration_minutes=scheduled_duration_minutes,
            )
            rescheduled_tasks.append(rescheduled_task)

        logger.info(f"Parsed {len(rescheduled_tasks)} rescheduled tasks")
        return rescheduled_tasks
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error parsing LLM response: {e}")
        return None


def optimize_schedule(
    model="llama2:latest", data_dir="data"
):  # specify 'llama2:latest'
    logger.info("Starting schedule optimization")

    # Check if the model is available
    if not is_ollama_model_available(model):
        logger.error(f"Ollama model '{model}' is not available.")
        return None

    # Load data from CSV files
    tasks = load_tasks_from_csv(data_dir=data_dir)
    calendar_entries = load_calendar_entries_from_csv(data_dir=data_dir)
    preferences = load_preferences_from_csv(data_dir=data_dir)

    if tasks is None or calendar_entries is None or preferences is None:
        logger.error("Failed to load data from CSV files.")
        return None

    # Create Calendar object and TaskAppUserPreferences object
    calendar = Calendar(id="1", name="Main Calendar", appointments=calendar_entries)
    task_scheduler_input = TaskSchedulerInput(
        tasks=tasks, calendar=calendar, preferences=preferences
    )

    # Create prompt
    prompt = create_prompt(task_scheduler_input)
    logger.debug(f"Generated prompt: {prompt}")

    # Query LLM
    response_text = query_ollama(prompt, model=model)
    if response_text is None:
        logger.error("Failed to get response from Ollama.")
        return None

    # Parse response into Python objects
    schedule = parse_schedule(response_text)

    logger.info("Schedule optimization completed")
    return schedule


def main():
    logger.info("Starting Task Scheduling Optimization")

    # Run the workflow and print results
    optimized_schedule = optimize_schedule(
        model="llama2:latest", data_dir="data"
    )  # specify 'llama2:latest'

    if optimized_schedule:
        print("Optimized Schedule:")
        for task in optimized_schedule:
            print(task)
    else:
        print("No schedule generated.")

    logger.info("Task Scheduling Optimization completed")


if __name__ == "__main__":
    logger.add(
        "ollama_manager.log", rotation="500 MB", level="INFO"
    )  # Add logging to a file
    main()
