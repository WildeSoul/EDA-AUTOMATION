# 🎮 Live Demo & Presentation Guide — Pro v4.0

This guide provides presenter scripts, recruiter pitches, resume descriptions, and step-by-step walkthroughs to showcase the **Agentic AI Data Scientist Platform (v4.0)**.

---

## 🏃 Step-by-Step Live Demo Scenario

### Step 1: Ingest & Profile (Titanic v1)
1. Navigate to the dashboard: `http://localhost:8000`.
2. Scroll to **Quick Demo Datasets** and click **Titanic Survival**.
3. Inspect the **Data Inspection** tab:
   - Check the **Data Health Score** (Grade B, ~84/100).
   - Check the **Semantic Typing** table (columns classified as `identifier`, `numeric`, or `binary_target`).
   - Note that **Survived** was correctly detected as the target.

### Step 2: Formulate Plan & Trigger HITL Approval Gate
1. Navigate to the **Autonomous Trace** tab.
2. In the query box, input:
   > *"Run a complete analysis to identify factors driving survival, check for outliers, and train a predictive classifier."*
3. Click **Generate Execution Plan**.
4. Review the estimated execution parameters. Since the generated plan contains multiple steps (Schema, Quality, Hypothesis, EDA, ML model) and the estimated runtime exceeds **30 seconds**, the **Human-in-the-Loop (HITL)** gate will trigger.
5. Click **Execute Plan**.
6. Observe that the pipeline transitions to **Awaiting Approval** and pauses execution.
7. Click the **Confirm Plan** button in the UI.
8. Watch the WebSocket stream update the nodes (`schema` ➔ `quality` ➔ `hypothesis` ➔ `eda` ➔ `ml` ➔ `insight`) with active spinners, model badges, and execution latencies in real time.
9. Note that independent branches (like `schema` and `quality`) execute in parallel.

### Step 3: Verify Version Lineage & Memory Search
1. Navigate to the **Memory & Knowledge** page.
2. Review the **Dataset Version Control & Lineage** table:
   - You will see the Titanic dataset listed.
3. In the **Semantic Memory Lookup** panel, search for: `passenger survival factors`.
4. Review the returned match cards, noting the **Similarity Score** and **Match Reasons** (e.g. `"same target type: binary_target"`).

### Step 4: Inspect Observability & Telemetry
1. Navigate to the **Observability & Analytics** tab in the sidebar.
2. Review the real-time execution statistics:
   - **Overall Success Rate**: Successful runs vs total executions.
   - **Parallel Speedup**: The percentage duration reduction achieved via parallel step scheduling (typically ~20%).
   - **Memory Rejection & Acceptance Rates**: Confirms that the $0.65$ confidence gate successfully filtered low-confidence memory injections.
   - **Agent Diagnostics**: Breakdown table of average durations, run counts, and retry frequencies.
   - **Graph Version Distribution**: Confirms which StateGraph compiles were invoked.

---

## 💼 Recruiter Pitch & Portfolio Copy

### 1-Minute Elevator Pitch
> *"I built an Agentic AI Data Scientist Platform that replaces simple static notebook scripts with a compiled LangGraph StateGraph orchestrator. It takes a raw dataset, extracts semantic features, pauses for human approval if the plan is complex, and schedules independent specialist agents in parallel over a thread pool—saving 20% in execution latency. State persistence is managed via thread-safe SQLite checkpoints and local ChromaDB memory with time-decayed fusion ranking. I also built a real-time observability console to track agent success rates, parallel speedup gains, and memory quality rejection rates. The entire codebase is verified by a 43-case pytest suite running 100% locally."*

### Resume Description Bullet-Points
- **Agentic Orchestration**: Built an autonomous multi-agent data scientist platform using a compiled **LangGraph StateGraph** runtime, scheduling parallel agent executions that reduced overall pipeline latency by **20%**.
- **Human-in-the-Loop Integration**: Implemented a **HITL node** using `NodeInterrupt` checkpointers to halt graph execution on complex plans ( rtime >30s) or low retrieval scores (<70%), enabling manual approval/resume events.
- **SQLite Checkpointing**: Programmed a thread-safe, custom **SQLite checkpointer** storing JSON state buffers across database tables to support session resumption and recovery from server restarts.
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

The entire system runs 100% locally on Ollama (Mistral Nemo + Nomic Embed) and is verified by a 43-case test suite.

#AI #GenerativeAI #LangGraph #DataScience #MachineLearning #SoftwareEngineering
```

---

## 🎤 Capstone Presentation Demo Script

### Presenter Script (Time: 3-5 Minutes)

| Time | Slide/Screen | Talk Script |
| :--- | :--- | :--- |
| **0:00 - 0:45** | **Architecture Diagram** | "Good morning/afternoon. Today I am presenting my Agentic AI Data Scientist Platform. Standard AI data systems run static scripts that break easily on unexpected inputs. My platform introduces an agentic workflow: we ingest data, build a semantic typing profile, query a local memory store, generate an execution DAG, and run it using a compiled LangGraph StateGraph." |
| **0:45 - 2:00** | **Live Upload & HITL** | "Let's run a demo. I will click the Titanic Survival dataset. If I ask a complex question, the Planner Agent constructs a topological plan. Because the plan contains multiple steps, the Human-in-the-Loop gate immediately intercepts execution, saving a checkpoint. As the presenter, I can inspect the steps, click 'Confirm', and resume. The engine now schedules steps like Schema Profiling and Data Quality checks in parallel, decreasing runtime by 20%." |
| **2:00 - 3:00** | **Memory & Versioning** | "Next, let's look at the Memory page. If we modify the dataset and re-upload, the system registers it as 'Version 2' and links it back to 'Version 1' via schema similarity. When we search for past findings, the vector store ranks memories by similarity, reuse counts, and time-decay, rejecting low-confidence inputs (<0.65) to protect agents from prompt contamination." |
| **3:00 - 4:00** | **Observability Console** | "Finally, let's visit the Observability Dashboard. This console monitors the platform's health. We can inspect overall success rates, measured parallel speedup gains, memory acceptance vs rejection ratios, and graph version counts. The entire application is verified by a 43-case test suite. Thank you, and I am open to any questions." |
