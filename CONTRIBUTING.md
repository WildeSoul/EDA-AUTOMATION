# 🤝 Contributing to the Agentic AI Data Scientist Platform

Thank you for your interest in contributing to the **Agentic AI Data Scientist Platform**! This guide outlines the setup procedures, testing protocols, and core architectural rules required to expand and maintain the codebase.

---

## 🛠️ Developer Setup & Guidelines

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/eda-automation.git
   cd eda-automation
   ```
2. **Virtual Environment**: Set up a virtual environment running Python 3.10, 3.11, or 3.12:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```
3. **Local LLM Settings**: Install [Ollama](https://ollama.com/) and preload the required embedding and reasoning models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral-nemo
   ```

---

## 🧪 Testing Protocol

The codebase is strictly validated by a robust unit and integration test suite using `pytest`. Any new features, endpoints, or agents must be covered by tests.

### Running Tests
- **Run the full suite**:
  ```bash
  pytest
  ```
- **Run specific test files**:
  ```bash
  pytest tests/test_sqlite_concurrency.py
  pytest tests/test_langgraph.py
  ```

### Writing Tests
- Place tests in the `tests/` directory following the `test_*.py` naming convention.
- Ensure integration tests use Mock / deterministic LLM fallbacks (provided in the testing configs) to prevent network delays and flaky runs.

---

## 🔒 SQLite & Concurrency Thread-Safety Rules

Because the platform schedules agent steps in parallel using a `ThreadPoolExecutor`, several threads will attempt to read and write to the database (`sessions.db`) at the same time. Follow these rules to avoid database locks:

1. **Enable WAL Mode & normal Sync**: All connections must execute the following pragmas:
   ```sql
   PRAGMA journal_mode = WAL;
   PRAGMA synchronous = NORMAL;
   PRAGMA foreign_keys = ON;
   ```
2. **Set Busy Timeout**: Instantiate all `sqlite3.connect` instances with `timeout=30.0` to permit busy-waits under concurrent writes.
3. **Thread Safety**: Set `check_same_thread=False` when instantiating database checkpointers or state connections, and protect shared connections with a thread-reentrant lock (`threading.RLock()`).
4. **No Shared Connections**: Never share a single, open connection object across threads without locking. Keep connections scoped to the transaction lifetime or wrap connection calls.

---

## 🤖 Extending Specialist Agents

The core worker agents are located in the `agents/` directory (e.g. `planner_agent.py`, `worker_agents.py`). To create a new specialist agent:

1. **Create the Agent Class**: Inherit from the base agent configurations.
2. **Define Schema Prompting**: Keep LLM inputs and outputs cleanly separated, and validate output formats using JSON schemas.
3. **Register in StateGraph**: Add the node and transition routes in [langgraph_engine.py](file:///c:/Users/laxmi/EDA%20automation/core/langgraph_engine.py) to integrate your agent into the core state machine.
4. **Update the UI**: Integrate rendering widgets for the new agent inside [sections.py](file:///c:/Users/laxmi/EDA%20automation/ui/sections.py) or [app.py](file:///c:/Users/laxmi/EDA%20automation/app.py).
