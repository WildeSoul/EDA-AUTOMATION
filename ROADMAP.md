# 🗺️ Platform Roadmap & Milestones

This document highlights the historical progression of the AI Data Scientist platform and plans out upcoming feature sprints.

---

## 🏁 Completed Milestones

### Phase 1 — Dataset Intelligence & Core Agents
- **Semantic Typing**: Fast heuristic column classification + LLM refinement.
- **Relational Store**: Thread-safe SQLite database for session Parquet, chat history, and agent output persistence.
- **DuckDB Integration**: compilation of Natural Language queries into executed DuckDB SQL.
- **Worker Agents**: Created modular worker classes (`SchemaAgent`, `QualityAgent`, `HypothesisAgent`, `EDAAgent`, `InsightAgent`).

### Phase 1.5 — Hardening & Caching
- **Benchmarking Suite**: Validated 30 unit tests and 100% accurate evaluations on Titanic, Churn, and Credit datasets.
- **Offline Cache**: MD5-based query caching layer to prevent repeated LLM calls.
- **Fallback Models**: Ordering of models (DeepSeek ➔ Gemma ➔ Qwen ➔ Local Mistral-Nemo) for cloud/offline availability.

### Phase 2 — Planner Agent & Execution Engine
- **Planner Agent**: Dynamic formulation of topologically sorted execution plans based on user questions.
- **Topological Executor**: Dependency sorting and dynamic output-to-input variable mapping.
- **WS Trace Streaming**: WebSockets API that streams step updates (Timing, Latency, Models) to an interactive execution DAG trace in the UI.

### Phase 3 — Memory & Knowledge Layer
- **ChromaDB Integration**: Persistent collections for profiling context.
- **Score Fusion Ranking**: Fusion of similarity, confidence, reuse, and recency to sort memory.
- **Memory Confidence Gate**: Configurable threshold (`0.65`) preventing low-relevance memory injection.
- **Lineage Graphs**: Automated tracking of dataset updates and relationships in SQLite.

---

## 🚀 Upcoming Phases

### Phase 4 — LangGraph Migration (Next Step)
- **StateGraph Core**: Replace the custom topological execution engine with LangGraph's `StateGraph`.
- **Node Execution**: Wrap existing agents as LangGraph nodes.
- **Conditional Routing**: Support conditional transitions (e.g. if quality tests fail, branch to cleaning node).
- **Parallel execution**: Execute independent nodes concurrently in the graph.
- **Node-level checkpoints**: Save intermediate states to SQLite to allow step-level rollbacks.

### Phase 5 — MLOps, Drift & Leakage Detection
- **Data Drift Detection**: Implement Kolmogorov-Smirnov and Population Stability Index (PSI) tests to flag distribution drift.
- **Leakage Guards**: Check for target leakage and features that correlation-leak label signals.

### Phase 6 — AutoML Scientist
- **Pipeline Auto-generation**: Generate end-to-end training code using scikit-learn/XGBoost.
- **Feature Importance**: Auto-compute SHAP feature importances.
