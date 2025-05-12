**`task-optimiser-llm-public`**

# Task Optimiser LLM

**Task Optimiser LLM** is a Python project for advanced task scheduling and optimisation, leveraging both numerical algorithms and large language models (LLMs). This repository is designed for developers and researchers interested in hybrid approaches to complex scheduling, validation, and optimisation workflows.

## Status

![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)

*This is an early-stage project. The API and features are subject to change as we refine the implementation and add new capabilities. This project is a limited and incomplete public preview of our internal work at DataBooth.*

## Features

- **Flexible Task Scheduling:** Optimise task allocation and sequencing using robust numerical methods.
- **Extensible Data Models:** Integrate with JSON and iCalendar for interoperability.
- **Modular Architecture:** Easily extend or swap optimisation and validation components.
- **LLM Integration:** Prepare for next-gen AI-driven validation and optimisation.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for dependency management)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/DataBooth/task-optimiser-llm-public.git
cd task-optimiser-llm-public
uv sync
```

### Usage

The main optimisation workflow is in `src/`. Example usage:

```python
from src.optimiser import schedule_tasks

tasks = [...]  # Your task definitions here
schedule = schedule_tasks(tasks)
print(schedule)
```

For detailed examples, see the [examples](./data/) directory.

## Project Structure

```
.
├── data/           # Example data and sample schedules
├── src/            # Core optimisation and scheduling logic
├── README.md
├── pyproject.toml
└── uv.lock
```

## Roadmap & Next Steps

### Coming Soon: Agentic LLM-based Validation

We are actively developing a parallel, agentic LLM-based validation component using the [pydanticAI](https://ai.pydantic.dev) library. This will:

- **Validate Schedules with LLMs:** Provide an orthogonal approach to numerical optimisation by using a language model agent to review, explain, and validate scheduling outputs.
- **Explainability:** Offer natural language feedback and suggestions for schedule improvements.
- **Hybrid Validation:** Enable users to compare and combine numerical and AI-driven validation for robust, trustworthy results.

## Contributing

We welcome contributions! Please open issues or submit pull requests for bug fixes, enhancements, or new features.

## License

Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---

**Questions or suggestions?**  

Open an issue or contact [github@databooth.com.au](mailto:github@databooth.com.au).
