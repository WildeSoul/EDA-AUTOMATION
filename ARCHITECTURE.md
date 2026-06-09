# 🏛️ System Architecture — Pro v4.0

This document details the underlying architecture of the **Agentic AI Data Scientist Platform (v4.0)**, explaining the core components, database ER schemas, compiled LangGraph StateGraph topology, memory fusion scoring, and observability metrics.

---

## 1. System Topology

The platform integrates a FastAPI web server, a multi-threaded parallel Execution Engine powered by LangGraph, a persistent local ChromaDB vector memory space, a relational SQLite session store, and a premium observability console.

```mermaid
graph TD
    UI[Web Console Dashboard] -->|WebSockets: Telemetry & Traces| API[FastAPI Server]
    UI -->|REST: Plan / Execute / Approve / Analytics| API
    
    API -->|Register dataframes| DuckDB[(DuckDB Analytics)]
    API -->|Save state checkpointer| Checkpointer[SQLite Checkpointer]
    Checkpointer -->|Thread-safe serialization| SQLite[(SQLite sessions.db)]
    
    subgraph "LangGraph StateGraph Engine"
        API -->|Compile & Run| StateGraph[StateGraph Machine]
        StateGraph -->|Conditional Memory Routing| Memory[Memory Agent]
        Memory -->|Semantic Search| Chroma[(ChromaDB Vector Store)]
        
        StateGraph -->|DAG Selection| Planner[Planner Agent]
        StateGraph -->|Human Approval Gate| HITL[HITL Gate Node]
        
        StateGraph -->|Multi-threaded Parallel Run| Executor[Parallel executor_node]
        Executor -->|Concurrently executing steps| Workers[Schema, Quality, Hypothesis, EDA, Insight Agents]
        
        StateGraph -->|Fallback diagnostic| Recovery[Recovery Node]
    end
```

---

## 2. Database & Schema Architectures

The database is built on SQLite (for structured metadata, lineages, execution states, and checkpointer steps) and ChromaDB (for time-decayed vector database memory).

### 2.1 SQLite Schema (Relational & Lineage DB)

```mermaid
erDiagram
    sessions {
        TEXT sid PK
        TEXT filename
        TEXT created_at
        INTEGER row_count
        INTEGER col_count
        TEXT intelligence
        REAL quality_score
        TEXT quality_grade
    }
    dataframes {
        TEXT sid PK
        BLOB data
    }
    chat_history {
        INTEGER id PK
        TEXT sid FK
        TEXT role
        TEXT content
        TEXT agent_name
        TEXT created_at
    }
    agent_results {
        INTEGER id PK
        TEXT sid FK
        TEXT agent_name
        TEXT result
        TEXT created_at
    }
    execution_plans {
        TEXT sid PK
        TEXT plan_json
        TEXT state_json
        TEXT updated_at
    }
    dataset_versions {
        TEXT fingerprint PK
        TEXT filename
        INTEGER version
        TEXT created_at
    }
    dataset_links {
        TEXT source_fingerprint PK
        TEXT target_fingerprint PK
        TEXT relationship_type PK
        REAL similarity_score
        TEXT created_at
    }
    checkpoint_history {
        TEXT thread_id PK
        TEXT checkpoint_ns PK
        TEXT checkpoint_id PK
        TEXT parent_checkpoint_id
        TEXT checkpoint_type
        BLOB checkpoint_bin
        TEXT metadata_type
        BLOB metadata_bin
        TEXT created_at
    }
    checkpoint_blobs {
        TEXT thread_id PK
        TEXT checkpoint_ns PK
        TEXT channel PK
        TEXT version PK
        TEXT type
        BLOB blob
    }
    checkpoint_writes {
        TEXT thread_id PK
        TEXT checkpoint_ns PK
        TEXT checkpoint_id PK
        TEXT task_id PK
        INTEGER idx PK
        TEXT channel
        TEXT value_type
        BLOB value_bin
        TEXT task_path
    }
    
    sessions ||--|| dataframes : "stores parquet"
    sessions ||--o{ chat_history : "logs messages"
    sessions ||--o{ agent_results : "persists outputs"
    sessions ||--|| execution_plans : "tracks DAG"
    sessions ||--o{ checkpoint_history : "checkpoint saves (thread_id = sid)"
    sessions ||--o{ checkpoint_blobs : "stores state variables"
    sessions ||--o{ checkpoint_writes : "logs intermediate step writes"
```

### 2.2 ChromaDB Collections (Vector Knowledge Base)

1. **`datasets` Collection**:
   - *Document*: Generated text summary describing features, data shapes, and inferred domain.
   - *Metadata*: `sid`, `fingerprint`, `filename`, `dataset_type`, `domain`, `version`, `row_count`, `col_count`, `quality_score`, `reuse_count`, `created_at`.
2. **`insights` Collection**:
   - *Document*: Synthesized reports and patterns.
   - *Metadata*: `sid`, `dataset_fingerprint`, `agent`, `title`, `confidence`, `reuse_count`, `created_at`.
3. **`hypotheses` Collection**:
   - *Document*: Formulated hypothesis statement.
   - *Metadata*: `sid`, `dataset_fingerprint`, `hypothesis`, `status`, `evidence_score`, `confidence`, `reuse_count`, `created_at`.
4. **`cleaning_actions` Collection**:
   - *Document*: Transformations and missing data imputation logs.
   - *Metadata*: `sid`, `dataset_fingerprint`, `column`, `action_type`, `created_at`.

---

## 3. LangGraph Execution Lifecycle

The compiled state machine orchestrates the execution flow from ingestion through memory injection, planning, human confirmation checkpoints, concurrent task execution, and recovery.

```mermaid
sequenceDiagram
    autonumber
    User->>UI: Submit Question / Analysis Request
    UI->>API: POST /api/sessions/{sid}/execute
    API->>StateGraph: compile & run(AgentState, config)
    
    StateGraph->>StateGraph: memory_router(state)
    alt Need Memory = Yes
        StateGraph->>Memory Agent: Retrieve cross-session memories
        Memory Agent->>ChromaDB: Query collections
        ChromaDB-->>Memory Agent: Return profiles & score fusion metrics
        Memory Agent-->>StateGraph: Inject Advisory Memory block
    end
    
    StateGraph->>Planner Agent: Generate topological execution plan
    StateGraph->>StateGraph: approval_gate(state)
    
    alt estimated_time > 30s OR memory_confidence < 0.70
        StateGraph->>Pre Approval Node: Commit status='awaiting_approval'
        StateGraph->>Approval Node: Raise NodeInterrupt (Halt Graph)
        StateGraph-->>API: Return paused execution state
        API-->>UI: Prompt user for plan confirmation
        User->>UI: Click CONFIRM plan
        UI->>API: POST /api/sessions/{sid}/approve
        API->>StateGraph: Resume checkpoint (approval_granted=True)
    end
    
    StateGraph->>Parallel Executor: execute_single_step() in parallel branches
    Parallel Executor->>WS: Stream live step_started / step_completed events
    WS-->>UI: Render progress spinners & timings in Trace Panel
    
    alt Any executor step fails
        StateGraph->>Recovery Node: Run fallback diagnostic insights
    end
    
    StateGraph->>ChromaDB: Auto-persist report insights/hypotheses/cleanups
    StateGraph-->>API: Return final StateGraph variables
    API-->>UI: Render report & refresh Observability Dashboard
```

---

## 4. Platform Analytics & Instrumentation

The dashboard aggregates database execution traces to render telemetry calculations without altering database models:
- **Parallel Speedup**: Calculated by comparing total sum of step durations with parallel executor node duration.
- **Retry Frequency**: Monitors step failures and tracks exponential backoff retries.
- **Memory Quality**: Evaluates ChromaDB vector lookups, computing average retrieval confidence and rejection rate based on the `0.65` threshold gate.
- **Deployment Auditing**: Visualizes version distribution counts of compiled `graph_version` configurations.
