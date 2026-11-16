# 📜 Changelog & Release Notes

This log tracks release histories, feature increments, and bug fixes for the **Agentic AI Data Scientist Platform** leading up to version **v4.0**.

---

## [v4.0-agentic-ai-data-scientist] — 2026-06-09
### Added
- **LangGraph Migration**: Core workflow rewritten from custom execution loops to a compiled LangGraph `StateGraph` machine.
- **SQLite Concurrency & WAL Hardening**: Enabled WAL (Write-Ahead Logging) mode, `check_same_thread=False`, and `30.0s` connection timeouts across checkpointers and session stores.
- **SaaS UI Redesign**: Fully restructured the dashboard navigation to a clean SaaS workflow sidebar (`🏠 Home`, `📂 Datasets`, `🔬 Analysis`, `🧠 Agent System`, `⚙️ Settings`).
- **Ingestion Landing Workspace**: Implemented drag-and-drop file uploader, cataloged demo datasets, and a Recent Sessions panel on the main page.
- **Integration Test Suite**: Added concurrent read/write stress integration test validating zero database locks under parallel operations.

### Changed
- Centralized model settings and engineering controls under the `⚙️ Settings` page.
- Wrapped database transactions in `SessionStore` inside reentrant locks (`threading.RLock`) to guarantee multi-threaded stability.
- Collapsed low-level agent hops and telemetry diagnostics under expandable Summaries to improve user interface readability.

### Fixed
- Fixed database conflicts and locks during high-frequency parallel agent executions.

---

## [v3.0-semantic-memory-layer] — 2026-05-15
### Added
- **Local ChromaDB Integration**: Integrated semantic vector space utilizing Nomic text embeddings.
- **Score Fusion Ranking**: Implemented multi-metric fusion formulas (similarity, quality confidence, log-reuse frequency, and time-decay) to sort retrieved memories.
- **Memory Confidence Gate**: Implemented a `0.65` threshold check to prevent prompt contamination from low-relevance past datasets.
- **Lineage Graphs**: Added structural metadata schemas in SQLite to link similar datasets across versions.

---

## [v2.0-planner-agent-dag] — 2026-04-10
### Added
- **Multi-Agent Planner**: Designed a Planner Agent generating topologically sorted execution DAGs based on custom user analytical questions.
- **ThreadPoolExecutor Nodes**: Programmed concurrent workers execution inside steps, reducing latency by ~20%.
- **WebSocket Streaming**: Created live telemetry WebSocket streams, updating agent trace DAGs, step timers, and LLM core badges.

---

## [v1.0-eda-automation-engine] — 2026-03-01
### Added
- **Dataset Intelligence Engine**: Created initial heuristic typing logic detecting targets, IDs, dates, and column structures.
- **Data Health Grading**: Built custom scoring criteria ($0-100$) evaluating duplicates, outliers, and missing cells into an A-F grade dashboard.
- **Specialty Labs**: Designed first-stage statistical summaries, normality tests, correlation heatmaps, and code sandboxes.
