# 🎮 Live Demo & Presentation Guide — Pro v4.0

This guide provides presenter scripts, recruiter pitches, resume descriptions, and step-by-step walkthroughs to showcase the **Agentic AI Data Scientist Platform (v4.0)**.

---

## 🏃 Step-by-Step Live Demo Scenario

### Step 1: Ingest & Profile (Titanic v1)
1. Navigate to the dashboard homepage: `http://localhost:8501`.
2. Under the **Try Demo Dataset** tab on the Home page, click **🚢 Titanic Survival**.
3. Observe the page automatically transitions to the **📂 Datasets** page:
   - Check the **Data Health Score Gauge** (Grade B, ~84/100).
   - Check the **Column Data Types** table (columns classified as `numeric`, `categorical`, or `binary_target`).
   - Observe that the target variable `Survived` is correctly auto-detected.

### Step 2: Formulate Plan & Trigger HITL Approval Gate
1. Navigate to the **🔬 Analysis** tab on the sidebar.
2. Observe the main interface renders two tabs: `💬 AI Conversational Chat & SQL` and `🧪 Specialty Labs`.
3. In the chat input, ask a complex question:
   > *"Analyze factors driving survival, check for outliers, and run a predictive model."*
4. Toggle to the **🧠 Agent System** sidebar page and select the **⚙️ LangGraph Orchestrator (DAG)** tab.
5. Review the plan. Since the generated plan contains multiple steps (Schema, Quality, Hypothesis, EDA, ML model) and the estimated runtime exceeds **30 seconds**, the **Human-in-the-Loop (HITL)** gate will trigger.
6. Click **Execute Node** and observe that the status pauses on **Awaiting Approval**.
7. Click the **Confirm Plan** button in the UI.
8. Watch the WebSocket stream update the nodes (`schema` ➔ `quality` ➔ `hypothesis` ➔ `eda` ➔ `ml` ➔ `insight`) with active spinners, model badges, and execution latencies in real time.
9. Note that independent branches (like `schema` and `quality`) execute in parallel.

### Step 3: Verify Version Lineage & Memory Search
1. On the **🧠 Agent System** page, toggle to the **🗂️ Memory & Knowledge** tab.
2. Review the **Dataset Schema & Version Lineage** panel:
   - You will see the Titanic dataset fingerprint registered as Version 1.
3. In the **Semantic Search** panel, search for: `passenger survival factors`.
4. Review the returned match cards, noting the **Fusion Score** and **Match Reasons** (e.g. `"similar_schema"`).

### Step 4: Inspect Observability & Telemetry
1. On the **🧠 Agent System** page, toggle to the **📊 Telemetry & Observability** tab.
2. Review the real-time execution statistics:
   - **Overall Success Rate**: Successful runs vs total executions.
   - **Parallel Execution Speedup**: The percentage duration reduction achieved via parallel step scheduling (typically ~20%).
   - **Memory Hit Rate**: Confirms that the $0.65$ confidence gate successfully filtered low-confidence memory injections.
   - Expand the **Show Technical Diagnostics & Audit Trail** expander to inspect average durations, run counts, and Graph versions.

---

## 💼 Recruiter Pitch & Portfolio Copy

### 1-Minute Elevator Pitch
> *"I built an Agentic AI Data Scientist Platform that replaces simple static notebook scripts with a compiled LangGraph StateGraph orchestrator. It takes a raw dataset, extracts semantic features, pauses for human approval if the plan is complex, and schedules independent specialist agents in parallel over a thread pool—saving 20% in execution latency. State persistence is managed via thread-safe SQLite checkpoints and local ChromaDB memory with time-decayed fusion ranking. I also built a real-time observability console to track agent success rates, parallel speedup gains, and memory quality rejection rates. The entire codebase is verified by a 44-case pytest suite running 100% locally."*

### Resume Description Bullet-Points
- **Agentic Orchestration**: Built an autonomous multi-agent data scientist platform using a compiled **LangGraph StateGraph** runtime, scheduling parallel agent executions that reduced overall pipeline latency by **20%**.
- **Human-in-the-Loop Integration**: Implemented a **HITL node** using `NodeInterrupt` checkpointers to halt graph execution on complex plans (runtime >30s) or low retrieval scores (<70%), enabling manual approval/resume events.
- **SQLite Concurrency & WAL Hardening**: Configured SQLite Write-Ahead Logging (WAL mode), normal synchronous writing, and thread safety flags to allow simultaneous multi-agent database reads and writes.
- **Observability Dashboard**: Developed an analytics dashboard calculating agent success rates, average durations, retry frequencies, and memory rejection metrics aggregated directly from DB execution logs.
- **Local Semantic Memory**: Integrated local **ChromaDB vector persistence** using `nomic-embed-text` with a multidimensional **Score Fusion** ranking formula incorporating vector similarity, quality score, reuse metrics, and exponential time-decay.

### LinkedIn Project Post
```text
🚀 I'm excited to share my latest project: The Agentic AI Data Scientist Platform (Pro v4.0)! 🧠📊

Moving beyond basic notebook analysis, I built a production-grade multi-agent platform designed to automate data profiling, cleaning, hypothesis generation, and modeling using local LLMs.

Core architectural highlights:
1️⃣ LangGraph StateGraph Engine: Migrated from sequential execution loops to a compiled state machine orchestrator.
2️⃣ Human-in-the-Loop (HITL) Gate: Automatically pauses graph execution on complex plans or low-confidence memory lookups, awaiting manual approval.
3️⃣ Parallel Executor: Runs independent worker agents concurrently over a ThreadPool, achieving a ~20% execution speedup.
4️⃣ SQLite State Checkpointer: Developed a thread-safe SQLite checkpoint saver allowing graph states to survive server crashes.
5️⃣ Time-Decayed Memory Fusion: Employs local ChromaDB storage to inject historical context ranked by similarity, recency decay, and quality confidence.
6️⃣ Real-time Observability: Built an analytics console visualizing agent success rates, parallel gains, and memory rejection rates.

The entire system runs 100% locally on Ollama (Mistral Nemo + Nomic Embed) and is verified by a 44-case test suite.

#AI #GenerativeAI #LangGraph #DataScience #MachineLearning #SoftwareEngineering
```

---

## 🎤 Capstone Presentation Demo Script

### Presenter Script (Time: 3-5 Minutes)

| Time | Slide/Screen | Talk Script |
| :--- | :--- | :--- |
| **0:00 - 0:45** | **Architecture Diagram** | "Good morning/afternoon. Today I am presenting my Agentic AI Data Scientist Platform. Standard AI data systems run static scripts that break easily on unexpected inputs. My platform introduces an agentic workflow: we ingest data, build a semantic typing profile, query a local memory store, generate an execution DAG, and run it using a compiled LangGraph StateGraph." |
| **0:45 - 2:00** | **Live Upload & HITL** | "Let's run a demo. I will click the Titanic Survival dataset on the Home Workspace. The app transitions to the Datasets tab where columns are typed and graded. If we ask a complex question under the Analysis chat, the Planner Agent constructs a topological plan. Toggle to the Agent System page; when we click execute, the Human-in-the-Loop gate immediately intercepts execution, saving a checkpoint. As the presenter, I can inspect the steps, click 'Confirm', and resume. The engine now schedules steps in parallel, decreasing runtime by 20%." |
| **2:00 - 3:00** | **Memory & Versioning** | "Next, let's look at the Memory & Knowledge tab. If we modify the dataset and re-upload, the system registers it as 'Version 2' and links it back to 'Version 1' via schema similarity. When we search for past findings, the vector store ranks memories by similarity, reuse counts, and time-decay, rejecting low-confidence inputs (<0.65) to protect agents from prompt contamination." |
| **3:00 - 4:00** | **Observability Console** | "Finally, let's visit the Telemetry & Observability tab. This console monitors the platform's health. We can inspect overall success rates, measured parallel speedup gains, memory acceptance vs rejection ratios, and graph version counts. Low-level logs are collapsed under expandable diagnostics. The entire application is verified by a 44-case test suite. Thank you, and I am open to any questions." |
