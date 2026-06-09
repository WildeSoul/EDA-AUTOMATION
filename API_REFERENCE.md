# 🔌 API Reference Guide — Pro v4.0

This document describes the FastAPI REST endpoints, request/response JSON schemas, and WebSocket progress streaming events for the Agentic AI Data Scientist Platform.

---

## 1. Ingestion & Sessions API

### `POST /api/upload`
Uploads a tabular file to initialize a session and compute dataset intelligence.
- **Request**: `multipart/form-data` with `file` field.
- **Response** (`200 OK`):
  ```json
  {
    "session_id": "8a9b2c3d",
    "filename": "titanic.csv",
    "columns": 12,
    "rows": 891,
    "numeric_cols": ["Age", "Fare"],
    "categorical_cols": ["Sex", "Embarked"],
    "date_cols": []
  }
  ```

### `GET /api/sessions`
Lists all historically saved sessions, sorted newest first.
- **Response** (`200 OK`):
  ```json
  {
    "sessions": [
      {
        "sid": "8a9b2c3d",
        "filename": "titanic.csv",
        "created_at": "2026-06-09 12:00:00",
        "row_count": 891,
        "col_count": 12,
        "quality_score": 88.5,
        "quality_grade": "B"
      }
    ]
  }
  ```

### `DELETE /api/sessions/{sid}`
Deletes a session, its original parquet dataframe, and all related execution plans/traces.
- **Response** (`200 OK`):
  ```json
  {
    "status": "deleted",
    "session_id": "8a9b2c3d"
  }
  ```

---

## 2. Planning & Execution API (LangGraph)

### `POST /api/sessions/{sid}/plan`
Generates and validates an execution plan for a user query.
- **Request**:
  ```json
  {
    "question": "What factors are strongly correlated with survival?"
  }
  ```
- **Response** (`200 OK`):
  ```json
  {
    "success": true,
    "plan": {
      "rationale": "Evaluating feature relationships and training classification models...",
      "estimates": {
        "estimated_steps": 3,
        "estimated_time_sec": 7.5,
        "estimated_llm_calls": 3
      },
      "steps": [
        {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}},
        {"id": "step_2", "agent": "quality", "depends_on": ["step_1"], "input_mapping": {"schema_result": "step_1"}},
        {"id": "step_3", "agent": "insight", "depends_on": ["step_2"], "input_mapping": {"quality_result": "step_2"}}
      ]
    }
  }
  ```

### `POST /api/sessions/{sid}/execute`
Runs the compiled LangGraph StateGraph pipeline, yielding execution telemetry.
- **Request**:
  ```json
  {
    "plan": { ... },
    "question": "What factors are strongly correlated with survival?"
  }
  ```
- **Response** (`200 OK` - Completed Execution):
  ```json
  {
    "success": true,
    "status": "completed",
    "trace": [
      {
        "step_id": "step_1",
        "agent": "schema",
        "description": "Profile dataset",
        "status": "completed",
        "duration_sec": 1.2,
        "model": "mistral-nemo",
        "error": null,
        "timestamp": "2026-06-09T12:00:05Z"
      }
    ],
    "outputs": {
      "step_1": { "agent": "schema", "success": true, "result": { ... } }
    },
    "requires_approval": false,
    "graph_version": "phase4_langgraph_v1",
    "graph_build_time": "2026-06-09T12:00:00Z",
    "node_telemetry": {
      "executor": { "attempts": 1, "durations": [1.0] }
    },
    "checkpoint_count": 2,
    "resume_events": []
  }
  ```
- **Response** (`200 OK` - Paused for Human-in-the-Loop Gate):
  ```json
  {
    "success": false,
    "status": "awaiting_approval",
    "trace": [],
    "outputs": {},
    "requires_approval": true,
    "graph_version": "phase4_langgraph_v1",
    "graph_build_time": "2026-06-09T12:00:00Z",
    "node_telemetry": {
      "pre_approval": { "attempts": 1, "durations": [0.01], "checkpoint_id": "ch_7a8f..." }
    },
    "checkpoint_count": 1,
    "resume_events": []
  }
  ```

### `POST /api/sessions/{sid}/approve`
Resumes a paused execution checkpoint, flagging `approval_granted=True`.
- **Request**:
  ```json
  {
    "plan": null
  }
  ```
- **Response**: Matches `/api/sessions/{sid}/execute` returns.

### `GET /api/sessions/{sid}/execution-state`
Loads the stored plan and StateGraph trace for a session.
- **Response**:
  ```json
  {
    "plan": { ... },
    "state": { ... }
  }
  ```

---

## 3. Memory & Observability API

### `GET /api/memory/datasets`
Retrieves historical dataset profiles and database version linkages.
- **Response**:
  ```json
  {
    "profiles": [
      {
        "fingerprint": "md5_hash",
        "filename": "titanic.csv",
        "version": 1,
        "created_at": "2026-06-09 12:00:00"
      }
    ]
  }
  ```

### `GET /api/memory/search`
Queries ChromaDB vector collection for similar sessions and insights.
- **Request parameters**: `q` (query string), `limit` (optional integer).
- **Response**:
  ```json
  {
    "results": [
      {
        "sid": "8a9b2c3d",
        "filename": "titanic.csv",
        "similarity": 0.88,
        "summary": "Passenger survival binary classification..."
      }
    ]
  }
  ```

### `GET /api/observability/metrics`
Aggregates execution plan data to calculate platform metrics.
- **Response**:
  ```json
  {
    "total_runs": 12,
    "overall_success_rate": 91.7,
    "parallel_speedup": 21.4,
    "memory_hit_rate": 75.0,
    "memory_metrics": {
      "acceptance_rate": 83.3,
      "rejection_rate": 16.7,
      "avg_confidence": 0.79,
      "total_checks": 24
    },
    "approval_metrics": {
      "requires_approval_count": 3,
      "approved_count": 3,
      "approval_rate": 100.0
    },
    "checkpoint_restores": 3,
    "retry_frequency": 8.3,
    "agent_metrics": [
      {
        "agent": "Schema",
        "runs": 12,
        "success_rate": 100.0,
        "avg_latency": 1.15,
        "retries": 0
      }
    ],
    "version_distribution": [
      { "version": "phase4_langgraph_v1", "runs": 12 }
    ],
    "recent_executions": [
      {
        "sid": "8a9b2c3d",
        "filename": "titanic.csv",
        "status": "completed",
        "graph_version": "phase4_langgraph_v1",
        "graph_build_time": "2026-06-09T12:00:00Z",
        "steps_count": 3,
        "checkpoint_count": 2,
        "timestamp": "2026-06-09T12:00:00Z"
      }
    ]
  }
  ```

---

## 4. WebSocket Streaming Event Contracts

WebSockets connect to `ws://localhost:8000/ws/{sid}`. The server streams execution events in real time.

### `step_started`
Fired when the ThreadPool begins executing an agent node.
- **Event Payload**:
  ```json
  {
    "type": "planner_event",
    "event": {
      "event": "step_started",
      "step_id": "step_1",
      "agent": "schema",
      "description": "Profile dataset"
    }
  }
  ```

### `step_completed`
Fired when a step finishes successfully.
- **Event Payload**:
  ```json
  {
    "type": "planner_event",
    "event": {
      "event": "step_completed",
      "step_id": "step_1",
      "agent": "schema",
      "duration": 1.25,
      "error": null
    }
  }
  ```

### `step_failed`
Fired when a step fails after all 3 retry attempts.
- **Event Payload**:
  ```json
  {
    "type": "planner_event",
    "event": {
      "event": "step_failed",
      "step_id": "step_1",
      "agent": "schema",
      "duration": 5.4,
      "error": "Temporary network connection issue"
    }
  }
  ```

### `awaiting_approval`
Fired when execution halts at the HITL approval gate.
- **Event Payload**:
  ```json
  {
    "type": "planner_event",
    "event": {
      "event": "awaiting_approval",
      "sid": "8a9b2c3d",
      "plan": { ... },
      "checkpoint_id": "ch_7a8f..."
    }
  }
  ```
