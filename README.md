# 🧠 Agentic AI Data Scientist Platform (Pro v4.0)

[![Tests Passing](https://img.shields.io/badge/Tests-43%20Passing-success?style=for-the-badge&logo=pytest&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/tests/test_observability.py)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph_StateGraph-blueviolet?style=for-the-badge&logo=openai&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/core/langgraph_engine.py)
[![SQLite Checkpointer](https://img.shields.io/badge/Checkpoint_Saver-Thread--safe_SQLite-blue?style=for-the-badge&logo=sqlite&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/data/checkpointer.py)
[![Observability](https://img.shields.io/badge/Observability-Dashboard_--_20%25_Speedup-orange?style=for-the-badge&logo=googlecloud&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/frontend/index.html)
[![LLM Engine](https://img.shields.io/badge/LLM_Engine-Ollama_--_Mistral_Nemo-red?style=for-the-badge&logo=linux&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/core/llm.py)
[![DuckDB](https://img.shields.io/badge/SQL_Engine-DuckDB-yellow?style=for-the-badge&logo=postgresql&logoColor=white)](file:///c:/Users/laxmi/EDA%20automation/data/engine.py)

An autonomous, production-ready **Agentic AI Data Scientist Platform** designed to ingest raw tabular datasets, semantically extract metadata profiles, pause for human approval gates, build topologically sorted parallel execution graphs, and persist execution states using SQLite checkpoints and ChromaDB memory.

Built with a compiled LangGraph state machine, thread-safe checkpoints, and a premium observability analytics console, this project demonstrates real-world agent orchestration, recovery fallbacks, and performance instrumentation.

---

## 🚀 Key Capabilities

- **LangGraph StateGraph Runtime**: Compiles agent executions into state machines with custom routing, including conditional memory routing and automated recovery pathways.
- **Human-in-the-Loop Node (HITL)**: Automatically interrupts and pauses execution when estimated runtimes exceed **30s** or memory confidence falls below **70%**, enabling manual plan reviews and resume actions.
- **Parallel Batch Execution**: Utilizes a multi-threaded pool to process independent steps concurrently, reducing execution latency by **~20%**.
- **Thread-safe SQLite Checkpointing**: Saves and restores state checkpoints of pipelines under `sessions.db`, allowing sessions to recover from server crashes or await long-running manual decisions.
- **Persistent Memory & Score Fusion**: Retrieves past insights, cleaning recommendations, and hypotheses using local ChromaDB vector stores, ranked using a time-decayed fusion formula.
- **Precision Dataset Intelligence**: Automatically detects target labels, identifiers, datetime categories, and data quality issues (100% typing accuracy benchmark).
- **System Observability & Analytics Dashboard**: Premium console rendering overall run metrics, parallel speedup percentages, memory acceptance/rejection rates, graph version distributions, and agent diagnostics.

---

## 🛠️ Quick Start & Setup

### Prerequisites
- Python 3.10 to 3.12 (Windows/MacOS/Linux)
- [Ollama](https://ollama.com/) running locally with the following models preloaded:
  - `nomic-embed-text` (For semantic vector memory)
  - `mistral-nemo:latest` or `qwen2.5-coder:7b` (For agent reasoning/planning)

### Installation
1. Clone the repository and navigate to the project root:
   ```bash
   cd "EDA automation"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend FastAPI server:
   ```bash
   python server.py
   ```
4. Access the web interface at: `http://localhost:8000`

---

## 🧪 Testing & Verification

Run the full pytest suite containing 43 verified unit and integration test cases covering agent execution, checkpointers, state graphs, parallel runs, and memory lookups:
```bash
python -m pytest
```

---

## 📖 Component Guides & Documentation

To deep dive into the engineering specifics, review the following guides:
- **[ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/ARCHITECTURE.md)**: Overall system layout, database ER schema, and component interactions.
- **[MEMORY_ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/MEMORY_ARCHITECTURE.md)**: ChromaDB vector collections, Score Fusion formulas, time-decay metrics, and target contamination safeguards.
- **[LANGGRAPH_ARCHITECTURE.md](file:///c:/Users/laxmi/EDA%20automation/LANGGRAPH_ARCHITECTURE.md)**: StateGraph topology design, SQLite checkpointers, parallel thread pool execution, and HITL gate parameters.
- **[API_REFERENCE.md](file:///c:/Users/laxmi/EDA%20automation/API_REFERENCE.md)**: FastAPI endpoints reference, schemas, and WebSocket progress progress events.
- **[DEPLOYMENT_GUIDE.md](file:///c:/Users/laxmi/EDA%20automation/DEPLOYMENT_GUIDE.md)**: Multi-stage Docker configurations, compose definitions, and environment variables.
- **[DEMO_GUIDE.md](file:///c:/Users/laxmi/EDA%20automation/DEMO_GUIDE.md)**: Recruiter pitches, capstone presentations, and presenter scripts matching the Titanic or Churn demo runs.
