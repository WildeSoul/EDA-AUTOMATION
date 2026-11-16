# 🧠 Agentic AI Data Scientist Platform (Pro v4.0)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph_StateGraph-blueviolet?style=for-the-badge&logo=langchain&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/core/langgraph_engine.py)
[![SQLite checkpointer](https://img.shields.io/badge/Checkpoint_Saver-SQLite_WAL-blue?style=for-the-badge&logo=sqlite&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/data/checkpointer.py)
[![ChromaDB](https://img.shields.io/badge/Memory_Layer-ChromaDB_Persistent-red?style=for-the-badge&logo=adguard&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/data/memory_store.py)
[![Streamlit](https://img.shields.io/badge/UI_Console-Streamlit_SaaS-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/app.py)
[![Tests Passing](https://img.shields.io/badge/Tests-44_Passing_--_100%25-success?style=for-the-badge&logo=pytest&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/tests/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/eda-automation/tests.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/your-username/eda-automation/actions)

An autonomous, production-ready **Agentic AI Data Scientist Platform** designed to ingest raw tabular datasets, semantically extract metadata profiles, pause for human approval gates, build topologically sorted parallel execution graphs, and persist execution states using SQLite checkpoints and ChromaDB memory.

Built with a compiled LangGraph state machine, thread-safe WAL checkpoints, and a premium observability analytics console, this project demonstrates real-world agent orchestration, recovery fallbacks, and performance instrumentation.

---

## 📸 Portfolio Showcase (Screenshots)

Once you capture screenshots of your dashboard views, place them in the `docs/screenshots/` directory to automatically display them here:

| Home Workspace | Dataset Intelligence |
| :---: | :---: |
| ![Home Workspace](docs/screenshots/home_page.png) <br> *Drag-and-drop uploads, demo selection & sessions* | ![Dataset Intelligence](docs/screenshots/datasets_view.png) <br> *Color-coded health score gauge & types* |

| Specialty Analytics | Agentic Execution Tracing |
| :---: | :---: |
| ![Specialty Labs](docs/screenshots/analysis_labs.png) <br> *Normality, correlations, anomalies & modeling* | ![Agent Trace](docs/screenshots/agent_dag.png) <br> *LangGraph step execution, approval gates & durations* |

---

## 💼 Recruiter & Portfolio Highlights

If you are reviewing this project for a technical position, here is the architectural complexity at a glance:
- **Topological Parallel Execution**: Evaluates agent step dependencies, running independent nodes concurrently over a `ThreadPoolExecutor` to reduce pipeline latency by **~20%**.
- **SQLite WAL Concurrency Hardening**: Hardened database connections against simultaneous multi-agent database writes by implementing Write-Ahead Logging (WAL mode), `30s` busy timeouts, and reentrant locks (`threading.RLock`).
- **Semantic Score Fusion**: Integrates a ChromaDB memory layer utilizing a custom fusion formula ranking past runs based on similarity ($50\%$), data quality ($20\%$), log reuse count ($20\%$), and exponential time-decay ($10\%$).
- **Human-in-the-Loop Interrupts**: Utilizes SQLite state checkpointers to pause execution and prompt the user if estimated agent step times exceed `30s` or retrieval score bounds fall below `0.70`.

---

## 🛠️ Quick Start & Setup

### Prerequisites
- Python 3.10 to 3.12 (Windows / MacOS / Linux)
- [Ollama](https://ollama.com/) running locally with these models preloaded:
  - `nomic-embed-text` (For semantic vector memory query lookup)
  - `mistral-nemo:latest` or `qwen2.5-coder:7b` (For core planning and reasoning)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/your-username/eda-automation.git
   cd eda-automation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web console application:
   ```bash
   python -m streamlit run app.py
   ```
4. Access the web interface at: `http://localhost:8501`

---

## 🧪 Testing & Verification

Run the full pytest suite containing 44 verified unit and integration test cases covering agent execution, checkpointers, state graphs, parallel runs, and memory lookups:
```bash
pytest
```

---

## 📖 Component Guides & Documentation

To deep dive into the engineering specifics, review the following guides:
- **[PROJECT_METRICS.md](file:///c:/Users/laxmi/EDA%20automation/PROJECT_METRICS.md)**: Structured resume highlights, technical benchmarks, and performance metrics.
- **[ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/ARCHITECTURE.md)**: Overall system layout, database ER schema, and component interactions.
- **[MEMORY_ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/MEMORY_ARCHITECTURE.md)**: ChromaDB vector collections, Score Fusion formulas, time-decay metrics, and target contamination safeguards.
- **[LANGGRAPH_ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/LANGGRAPH_ARCHITECTURE.md)**: StateGraph topology design, SQLite checkpointers, parallel thread pool execution, and HITL gate parameters.
- **[API_REFERENCE.md](file:///c:/Users/laxmi/EDA%20automation/API_REFERENCE.md)**: FastAPI endpoints reference, schemas, and WebSocket progress progress events.
- **[DEPLOYMENT_GUIDE.md](file:///c:/Users/laxmi/EDA%20automation/DEPLOYMENT_GUIDE.md)**: Multi-stage Docker configurations, compose definitions, and environment variables.
- **[DEMO_GUIDE.md](file:///c:/Users/laxmi/EDA%20automation/DEMO_GUIDE.md)**: Recruiter pitches, capstone presentations, and presenter scripts matching the Titanic or Churn demo runs.
- **[ROADMAP.md](file:///c:/Users/laxmi/EDA%20automation/ROADMAP.md)**: Project historical evolution and future phases.
- **[CONTRIBUTING.md](file:///c:/Users/laxmi/EDA%20automation/CONTRIBUTING.md)**: Local developer setup, testing parameters, and SQLite concurrency safety rules.
- **[CHANGELOG.md](file:///c:/Users/laxmi/EDA%20automation/CHANGELOG.md)**: System release notes and iteration logs.
