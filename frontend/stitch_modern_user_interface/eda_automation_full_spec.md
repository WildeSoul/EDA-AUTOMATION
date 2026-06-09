# AI Data Analyst Pro — Complete Application Specification
### For Frontend Rebuild (Google Stitch Handoff)

> **Current Stack**: Python + Streamlit (backend+UI) + Ollama LLM (local REST API)
> **Goal**: Replace Streamlit UI with a modern web frontend while keeping the Python backend as an API

---

## 1. APPLICATION OVERVIEW

**Name**: AI Data Analyst Pro
**Purpose**: Upload any CSV/Excel dataset → full automated EDA + AI-powered Q&A + ML modeling + report export
**Current Entry**: `streamlit run app.py`

### Core User Flow
```
Upload File → Data Inspection → Statistical Profiling → Feature Engineering
→ Visualizations → Advanced Stats → Advanced Analysis → Anomaly Detection
→ ML Modeling → Code Lab → AI Summary → AI Q&A Chat → Report Export
```

---

## 2. PROJECT STRUCTURE

```
EDA automation/
├── app.py                          # Main entry point (orchestrator)
├── requirements.txt                # Python dependencies
├── .env                            # API keys (NVIDIA_API_KEY)
│
├── config/
│   └── settings.py                 # ALL constants, model configs, timeouts, system prompts
│
├── core/
│   ├── llm.py                      # Ollama REST API communication layer
│   ├── context.py                  # Builds rich data context strings for LLM
│   └── agentic.py                  # Plan → Execute → Synthesize pipeline
│
├── analysis/
│   ├── features.py                 # Date extraction, interactions, skewness, imputation
│   ├── stats.py                    # Normality tests, chi-square, PCA, clustering
│   ├── anomaly.py                  # Isolation Forest, Z-Score, IQR multivariate
│   └── ml.py                       # Quick ML: RF, LR, XGBoost (classification + regression)
│
├── tools/
│   ├── code_sandbox.py             # Safe Python code execution sandbox
│   ├── nl_viz.py                   # Natural language → chart generation
│   └── report.py                   # HTML report export with embedded charts
│
└── ui/
    ├── styles.py                   # CSS injection (Inter font, dark theme, gradients)
    ├── sidebar.py                  # File upload, model selection, toggle options
    └── sections.py                 # ALL 12 UI section renderers (50KB, 1028 lines)
```

---

## 3. BACKEND API — Ollama REST Interface

The backend communicates with Ollama via local REST API at `http://localhost:11434`.

### 3.1 Available Endpoints (used by the app)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/generate` | POST | Send prompt, get response (sync or streaming) |
| `/api/tags` | GET | List installed models |
| `/api/generate` (keep_alive=0) | POST | Unload model from VRAM |

### 3.2 LLM Functions (`core/llm.py`)

```python
def query_ollama(model: str, prompt: str, system: str = "", timeout: int = 300) -> str
    """Send prompt to Ollama, return complete response text."""
    # POST /api/generate {model, prompt, stream: false, system, options: {num_ctx: 8192, num_gpu: 99}}
    # Returns: response text string
    # Errors: connection error, HTTP error, OOM detection

def stream_ollama(model: str, prompt: str, system: str = "") -> Generator[str]
    """Stream tokens from Ollama as a generator (for real-time chat)."""
    # POST /api/generate {model, prompt, stream: true, system, options}
    # Yields: individual tokens as they arrive
    # Used by: chat Q&A, consensus synthesis

def unload_ollama_model(model: str) -> None
    """Free VRAM by unloading a model."""
    # POST /api/generate {model, keep_alive: 0}

def get_ollama_models() -> list[str]
    """Fetch installed model names, sorted by preference ranking."""
    # GET /api/tags → returns model names sorted by PREFERRED_ORDER

def query_nvidia_api(prompt, system, api_key, model, timeout) -> str
    """Fallback: call NVIDIA's OpenAI-compatible API."""
    # POST https://integrate.api.nvidia.com/v1/chat/completions
```

### 3.3 Available LLM Models (10 total)

| # | Model ID | Type | Tier | Specialty |
|---|---|---|---|---|
| 1 | `gemma4:31b-cloud` | ☁️ Cloud | 1 — Frontier | Reasoning, multimodal, planning |
| 2 | `qwen3.5:cloud` | ☁️ Cloud | 1 — Frontier | Data analysis, reasoning |
| 3 | `kimi-k2.5:cloud` | ☁️ Cloud | 1 — Frontier | Multimodal agentic |
| 4 | `glm-5.1:cloud` | ☁️ Cloud | 1 — Frontier | Agentic engineering, code |
| 5 | `qwen3-coder-next:cloud` | ☁️ Cloud | 2 — Specialist | Code generation |
| 6 | `nemotron-3-super:cloud` | ☁️ Cloud | 2 — Specialist | MoE reasoning (120B, 12B active) |
| 7 | `nemotron-3-nano:30b-cloud` | ☁️ Cloud | 2 — Specialist | Fast 30B executor |
| 8 | `minimax-m2.7:cloud` | ☁️ Cloud | 2 — Specialist | Coding & productivity |
| 9 | `mistral-nemo:latest` | 💻 Local | 3 — Fallback | 12.2B local workhorse |
| 10 | `llama3.1:latest` | 💻 Local | 3 — Fallback | 8B lightweight |

### 3.4 Agentic Pipeline Models (for Plan→Execute→Synthesize)

| Role | Model | Purpose |
|---|---|---|
| `planner` | `gemma4:31b-cloud` | Decomposes question into 3-5 analytical sub-tasks |
| `executor` | `qwen3.5:cloud` | Executes each sub-task against data |
| `code_gen` | `qwen3-coder-next:cloud` | Generates Python code (Code Lab + NL→Viz) |
| `executor_local` | `mistral-nemo:latest` | Local fallback when cloud unavailable |

---

## 4. CONFIGURATION (`config/settings.py`)

```python
# Ollama connection
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_NUM_CTX = 8192          # Context window (tokens)
OLLAMA_GPU_LAYERS = 99         # GPU offloading layers

# Timeouts (seconds)
TIMEOUT_LOCAL = 300            # 5 min for local models
TIMEOUT_CLOUD = 120            # 2 min for cloud models
TIMEOUT_NVIDIA_API = 600       # 10 min for NVIDIA API fallback

# ML Defaults
ML_TEST_SIZE = 0.2             # 80/20 train-test split
ML_RANDOM_STATE = 42           # Reproducibility seed
ML_CV_FOLDS = 5                # Cross-validation folds

# Anomaly Detection Defaults
ANOMALY_CONTAMINATION_DEFAULT = 0.05   # 5% expected anomalies
ANOMALY_ZSCORE_THRESHOLD = 3.0         # 3σ threshold

# Code Sandbox — allowed Python builtins
SANDBOX_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "print", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
}
```

### System Prompt (for all AI responses)
The AI persona is an "elite data scientist with 20+ years experience." Key rules enforced:
1. Cite exact numbers (columns, values, p-values) — never vague
2. Show reasoning (WHY, not just WHAT)
3. Cross-validate findings
4. Provide actionable output with thresholds/code
5. Rate confidence (High/Medium/Low)
6. Translate to business impact
7. Think step-by-step

---

## 5. DATA INPUT & PROCESSING

### 5.1 File Upload
- **Formats**: CSV (`.csv`), Excel (`.xlsx`, `.xls`)
- **Loading**: `pd.read_csv()` or `pd.read_excel()` (uses openpyxl)
- **Caching**: Streamlit `@st.cache_data` — loads once, reuses on reruns

### 5.2 Column Classification (automatic)
```python
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
date_cols = detect_date_columns(df)  # auto-detects date-like strings
```

### 5.3 Feature Engineering Pipeline (`analysis/features.py`)

| Function | Input | Output | Description |
|---|---|---|---|
| `detect_date_columns(df)` | DataFrame | `list[str]` | Auto-detects date columns by parsing samples |
| `engineer_date_features(df, date_cols)` | DataFrame, date cols | `(df, new_features)` | Extracts: year, month, day, dayofweek, quarter, is_weekend |
| `engineer_interaction_features(df, numeric_cols, top_n=3)` | DataFrame, numeric cols | `(df, new_features)` | Creates multiplication & ratio features from top correlated pairs |
| `compute_skewness(df, numeric_cols)` | DataFrame, numeric cols | `DataFrame` | Flags features with \|skewness\| > 1, suggests transforms |
| `impute_missing(df, numeric, categorical)` | DataFrame, col lists | `(df, report_dict)` | Median for numeric, mode for categorical |

### 5.4 Data Context Builder (`core/context.py`)
```python
def build_data_context(df, uploaded_name, numeric_cols, categorical_cols,
                       engineered_features, imputation_report, skew_df,
                       date_cols, user_question) -> str
```
Builds a comprehensive text summary of the dataset for LLM consumption, including:
- Shape, column types, date ranges
- Descriptive statistics (describe())
- Categorical breakdowns (top 7 values per column)
- Top 5 positive & negative correlations
- Outlier counts (IQR method)
- Missing values & duplicates
- Engineered features list
- Imputation report
- Skewness flags
- Relevant rows matching user's question (keyword/number matching)

---

## 6. UI SECTIONS (12 Total)

### Section 1: Data Inspection & Health Check
**Function**: `render_data_inspection(df)`

| Component | Type | Data |
|---|---|---|
| Metrics row (5 cards) | KPI Cards | Rows, Columns, Duplicates, Missing Cells, Memory |
| Data Preview | Table | First 20 rows of DataFrame |
| Column Data Types | Table | Column name, type, non-null count, unique count |
| Missing Values | Table + Bar Chart | Columns with missing values, %, color-coded bar chart |

---

### Section 2: Statistical Profiling
**Function**: `render_statistical_profiling(df, numeric_cols, categorical_cols)`

| Tab | Content |
|---|---|
| Numerical Summary | describe() + skewness + kurtosis per column |
| Categorical Summary | Unique values, top value, top frequency, missing count |
| Full Describe | Full `df.describe(include="all")` transposed |

---

### Section 3: Feature Engineering (toggleable)
**Function**: `render_feature_engineering(df, numeric_cols, categorical_cols)`
**Returns**: `(df, engineered_features, imputation_report, skew_df, date_cols, numeric_cols)`

| Sub-feature | Description |
|---|---|
| Date Feature Extraction | Detects dates → creates year/month/day/dow/quarter/weekend |
| Missing Value Imputation | Median (numeric), Mode (categorical) with detailed report |
| Skewness Detection | Flags columns with \|skew\| > 1, suggests log/square transforms |
| Interaction Features | Top 3 correlated pairs → creates multiplication & ratio features |

---

### Section 4: Automated Visualizations
**Function**: `render_visualizations(df, numeric_cols, categorical_cols, date_cols, section_num)`

| Tab | Chart Type | Details |
|---|---|---|
| 📊 Distributions | Histogram + Box Plot + Violin | User selects column; 3 side-by-side plots |
| 📦 Categories | Horizontal Bar Chart | Top 15 categories with count labels |
| 🔥 Correlations | Heatmap + Table | Triangular heatmap (RdBu colormap), top 15 pairs table |
| 📈 Pairwise | Pair Plot | Seaborn pairplot with KDE diagonal, optional hue coloring |
| 📉 Time Series | Line Chart + Moving Average | Date column vs numeric column, rolling average overlay |

---

### Section 5: Advanced Statistical Tests (toggleable)
**Function**: `render_advanced_stats(df, numeric_cols, categorical_cols, section_num)`
**Returns**: `(pca_data, pca_idx)`

| Tab | Statistical Method | Output |
|---|---|---|
| 🔔 Normality | Shapiro-Wilk + D'Agostino | Per-column: stat, p-value, ✅/❌ normal, metric cards |
| 🔗 Chi-Square | Chi-square independence + Cramér's V | Pair-wise: chi², p-value, V, significance, strength |
| 📉 PCA | Principal Component Analysis | Variance bar chart, 2D scatter, component loadings table |
| 🎯 Clustering | KMeans + Elbow Method | Elbow chart, 2D cluster scatter (colored), cluster stats table |

---

### Section 6: Advanced Analysis
**Function**: `render_advanced_analysis(df, numeric_cols, categorical_cols, section_num)`
**Returns**: `target_col`

| Tab | Content |
|---|---|
| 🎯 Target Analysis | User selects target → distribution plots (numeric) or pie/bar chart (categorical) + metrics |
| 📊 Feature Importance | Correlation-based importance bars (color coded: red=strong, yellow=moderate, blue=weak) |
| 🔎 Outlier Deep Dive | Per-column IQR outlier detection, scatter + boxplot visualization, outlier row viewer |

---

### Section 7: Anomaly Detection (toggleable)
**Function**: `render_anomaly_detection(df, numeric_cols, section_num)`
**Returns**: `anomaly_count`

| Tab | Algorithm | Controls | Output |
|---|---|---|---|
| 🌲 Isolation Forest | sklearn IsolationForest (200 trees) | Contamination slider (0.01–0.20) | PCA scatter (red=anomaly, blue=normal), metrics, CSV download |
| 📏 Z-Score | Standard Z-Score | Threshold slider (2.0–5.0) | Per-column anomaly counts, anomalous rows table |
| 📦 IQR Multivariate | 1.5×IQR across multiple columns | None | Rows outlier in 2+ features simultaneously |

---

### Section 8: Quick ML Modeling (toggleable)
**Function**: `render_ml_modeling(df, numeric_cols, categorical_cols, section_num)`
**Returns**: `ml_results` dict

| Feature | Details |
|---|---|
| Auto task detection | If target has ≤10 unique values → classification, else → regression |
| Available models | Random Forest, Logistic/Linear Regression, XGBoost |
| Feature selection | Multiselect dropdown for features |
| Train/test split | 80/20, stratified for classification |
| Cross-validation | 5-fold CV with mean ± std |
| Metrics (classification) | Accuracy, F1-weighted, confusion matrix heatmap |
| Metrics (regression) | R², RMSE, actual vs predicted scatter |
| Feature importance | Bar chart from model.feature_importances_ or coef_ |

---

### Section 9: AI Code Lab (toggleable)
**Function**: `render_code_lab(df, numeric_cols, categorical_cols, selected_model, section_num)`

| Feature | Details |
|---|---|
| Input | Text area: natural language request |
| Model | Uses `qwen3-coder-next:cloud` (code generation specialist) |
| Processing | LLM generates Python code → sandboxed execution |
| Sandbox | Restricted builtins, only pd/np/plt/sns allowed |
| Output | Generated code viewer, stdout, result DataFrame, matplotlib figures |
| Safety | No file I/O, no os/sys/subprocess allowed |

---

### Section 10: AI Executive Summary (toggleable)
**Function**: `render_ai_summary(df, uploaded_name, ..., selected_model, section_num)`
**Returns**: `ai_summary` string

| Feature | Details |
|---|---|
| Auto-generates | Data Quality Grade (A-F), Top 3 Patterns, Top 3 Recommendations, Use Cases, Red Flags |
| Caching | Cached in `st.session_state` keyed by filename |
| Regenerate | Button to clear cache and regenerate |

---

### Section 11: AI-Powered Q&A Chat
**Function**: `render_qa_chat(df, ..., selected_model, available_models, ..., run_agentic, run_consensus)`

This is the main interactive feature with 3 modes + visualization detection:

#### Mode 1: Single Model Chat
- User types question → streamed response from selected model
- Previous conversation context included (last 6 messages)
- Structured response format: Direct Answer → Key Insights → Solution → Warnings → Follow-up → Confidence

#### Mode 2: Agentic Deep Analysis (toggleable)
```
Step 1/3 — Planning: planner model (gemma4:31b-cloud) decomposes question into 3-5 sub-tasks
Step 2/3 — Executing: executor model (qwen3.5:cloud) runs each sub-task against data
Step 3/3 — Synthesizing: planner model merges all results into executive-level answer
```
Each step is shown in expandable sections with full detail.

**Agentic Pipeline Functions** (`core/agentic.py`):
```python
def agentic_plan(question, data_context) -> (raw_plan_text, list_of_steps)
    # Uses planner model to decompose into 3-5 numbered analytical sub-tasks
    
def agentic_execute(steps, data_context) -> list[{step, result}]
    # Executes each sub-task using executor model, returns findings
    
def agentic_synthesize(question, plan, exec_results) -> str
    # Produces: Direct Answer, Statistical Evidence, Key Insights,
    # Data Quality & Caveats, Actionable Recommendations, Confidence Level
```

#### Mode 3: Multi-Model Consensus (toggleable)
- Queries up to 3 models in parallel
- Shows individual responses in expandable sections
- Synthesizes: agreements, disagreements, best combined answer

#### Auto Visualization Detection
```python
VIZ_KEYWORDS = {"plot", "chart", "graph", "visualize", "show me", "draw",
                "histogram", "scatter", "bar chart", "pie chart", "heatmap",
                "boxplot", "violin", "line chart", "trend", "distribution"}
```
If user question contains visualization keywords → generates chart via `qwen3-coder-next:cloud` → executes in sandbox → displays chart + code.

---

### Section 12: Report Export
**Function**: `render_report_export(df, ..., ai_summary, anomaly_count, ml_results, section_num)`

Generates a **standalone HTML report** with:
- Professional dark theme (gradient headers, stat cards, styled tables)
- Data overview metrics (rows, cols, missing, duplicates, memory, anomalies)
- Missing values breakdown table
- Descriptive statistics table
- Distribution charts (embedded as base64 PNG)
- Correlation heatmap (embedded as base64 PNG)
- Categorical summary
- ML model results (if trained)
- AI executive summary
- Footer with generation timestamp

---

## 7. SIDEBAR CONFIGURATION
**Function**: `render_sidebar()` → returns `dict`

### Controls

| Control | Type | Default | Key |
|---|---|---|---|
| File Upload | File uploader | — | `uploaded_file` |
| Model Selection | Dropdown | First in PREFERRED_ORDER | `selected_model` |
| Auto Feature Engineering | Checkbox | ✅ On | `run_feature_eng` |
| Advanced Statistical Tests | Checkbox | ✅ On | `run_advanced_stats` |
| Anomaly Detection | Checkbox | ❌ Off | `run_anomaly` |
| Quick ML Model | Checkbox | ❌ Off | `run_quick_ml` |
| AI Auto-Summary | Checkbox | ❌ Off | `run_auto_summary` |
| Agentic Deep Analysis | Checkbox | ❌ Off | `run_agentic` |
| Multi-Model Consensus | Checkbox | ❌ Off | `run_consensus` |
| Code Lab | Checkbox | ❌ Off | `run_code_lab` |

### Model Display
- Cloud models shown with ☁️ prefix
- Local models shown with 💻 prefix
- Caption changes based on selection

---

## 8. STYLING & DESIGN SYSTEM

### Current Theme (dark mode)
```css
Font: 'Inter' (Google Fonts) — weights 300-700
Background: Streamlit dark default (#0f1117)
Accent: #667eea → #764ba2 (gradient)
Borders: rgba(102, 126, 234, 0.15-0.2)
Cards: #1a1d29 background, #2a2d3a borders
Charts: #3498db (blue), #e74c3c (red), #2ecc71 (green), #f39c12 (yellow), #9b59b6 (purple)
```

### Visual Elements
- Gradient text on h1 headings
- Rounded borders on all components (8-12px)
- Hover lift effect on buttons (+box-shadow)
- Gradient download buttons
- Tab pills with rounded styling

---

## 9. DEPENDENCIES

```
streamlit>=1.30.0              # Current UI framework (to be replaced)
pandas>=2.0.0                  # Core data manipulation
numpy>=1.24.0                  # Numerical computing
matplotlib>=3.7.0              # Chart generation
seaborn>=0.12.0                # Statistical visualization
scipy>=1.10.0                  # Statistical tests
scikit-learn>=1.3.0            # ML models, PCA, clustering, scaling
xgboost>=2.0.0                 # Gradient boosting models
requests>=2.28.0               # HTTP client for Ollama API
python-dotenv>=1.0.0           # .env file loading
openpyxl>=3.1.0                # Excel file reading
```

---

## 10. KEY DATA STRUCTURES

### Sidebar Config Dict (returned by `render_sidebar()`)
```python
{
    "uploaded_file": UploadedFile | None,
    "selected_model": str,           # e.g. "gemma4:31b-cloud"
    "available_models": list[str],   # all installed models
    "run_feature_eng": bool,
    "run_advanced_stats": bool,
    "run_anomaly": bool,
    "run_quick_ml": bool,
    "run_auto_summary": bool,
    "run_agentic": bool,
    "run_consensus": bool,
    "run_code_lab": bool,
}
```

### ML Results Dict (returned by `train_quick_model()`)
```python
{
    "success": bool,
    "error": str | None,
    "task_type": "classification" | "regression",
    "model_name": str,
    "n_samples": int,
    "n_features": int,
    # Classification:
    "accuracy": float,
    "f1_score": float,
    "confusion_matrix": np.ndarray,
    "classification_report": dict,
    # Regression:
    "r2": float,
    "rmse": float,
    "mae": float,
    "y_test": np.ndarray,
    "y_pred": np.ndarray,
    # Both:
    "cv_scores": np.ndarray,
    "cv_mean": float,
    "cv_std": float,
    "feature_importance": pd.DataFrame,  # columns: Feature, Importance
}
```

### Code Sandbox Result Dict (returned by `execute_code_safely()`)
```python
{
    "success": bool,
    "stdout": str,                    # captured print output
    "result_df": pd.DataFrame | None, # if code assigns to 'result'
    "figures": list[plt.Figure],      # captured matplotlib figures
    "error": str | None,
}
```

### Anomaly Detection Results
```python
# Isolation Forest:
(result_df: DataFrame,            # original df + anomaly_label + anomaly_score columns
 anomaly_mask: np.ndarray,        # boolean array (True = anomaly)  
 scores: np.ndarray)              # decision function scores

# Z-Score:
(summary_df: DataFrame,           # columns: Column, Anomalies, % of Data, Max Z-Score
 anomalous_rows: DataFrame)       # rows where any feature exceeds threshold

# IQR Multivariate:
(summary_df: DataFrame,           # columns: Column, Outliers, Lower Bound, Upper Bound, % Outliers
 anomalous_rows: DataFrame)       # rows outlier in 2+ features + outlier_features_count column
```

---

## 11. CHAT SESSION STATE

```python
st.session_state.chat_history = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
]
```

The Q&A chat maintains conversation context. The last 6 messages are included in each prompt as `PREVIOUS CONVERSATION` context.

---

## 12. IMPORTANT BEHAVIOR NOTES FOR FRONTEND

1. **File upload is the gate**: Nothing renders until a file is uploaded. Landing page shows when no file.

2. **Section numbering is dynamic**: Sections auto-number based on which toggles are enabled.

3. **Model selection affects everything**: The selected model is used for all AI features (summary, Q&A, code lab uses code_gen override).

4. **Cloud vs Local distinction matters**: Cloud models get 120s timeout, local models get 300s. Cloud models display ☁️, local show 💻.

5. **Streaming is critical for chat**: The Q&A chat streams tokens in real-time via `stream_ollama()`. The frontend must support Server-Sent Events or WebSocket streaming.

6. **Charts are matplotlib/seaborn**: All charts are currently generated server-side as matplotlib figures. The frontend could either:
   - Keep server-side chart generation (return as base64 PNG)
   - Replace with client-side charting (D3.js, Chart.js, Plotly.js)

7. **Code sandbox is server-side only**: The Code Lab executes LLM-generated Python code on the server. This must remain server-side for security.

8. **Agentic pipeline has 3 visible steps**: Users see "Planning... → Executing N sub-tasks... → Synthesizing..." with expandable details at each step.

9. **Multi-model consensus queries up to 3 models**: Shows individual responses + synthesized consensus. Previous models are unloaded from VRAM between queries.

10. **HTML report is self-contained**: The exported report includes all CSS inline and charts as base64 data URIs — no external dependencies.

---

## 13. SUGGESTED FRONTEND ARCHITECTURE

### If building as a modern web app:

```
Frontend (React/Next.js/Vue)          Backend (FastAPI)
┌─────────────────────┐               ┌─────────────────────┐
│  File Upload         │───upload────→│  /api/upload         │
│  Dashboard           │──────────────│  /api/inspect        │
│  Charts (Plotly.js)  │──────────────│  /api/profile        │
│  Chat (WebSocket)    │──streaming──→│  /api/chat           │
│  Code Lab            │──────────────│  /api/code-lab       │
│  ML Controls         │──────────────│  /api/ml/train       │
│  Settings Panel      │──────────────│  /api/models         │
│  Report Download     │──────────────│  /api/report         │
└─────────────────────┘               └──────┬──────────────┘
                                             │
                                      ┌──────▼──────────────┐
                                      │  Ollama (localhost)  │
                                      │  :11434              │
                                      └─────────────────────┘
```

### Suggested API Endpoints to Create:

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /api/upload` | POST | Upload CSV/Excel, return dataset ID + metadata |
| `GET /api/inspect/{id}` | GET | Data inspection metrics + preview |
| `GET /api/profile/{id}` | GET | Statistical profiling data |
| `POST /api/features/{id}` | POST | Run feature engineering, return results |
| `GET /api/visualize/{id}` | GET | Get chart data (or base64 images) |
| `GET /api/stats/{id}` | GET | Advanced stats (normality, chi-sq, PCA, clusters) |
| `POST /api/anomaly/{id}` | POST | Run anomaly detection (params: method, threshold) |
| `POST /api/ml/train/{id}` | POST | Train ML model (params: target, features, model) |
| `POST /api/code-lab/{id}` | POST | Generate + execute code |
| `POST /api/chat/{id}` | POST/WS | Chat Q&A (streaming via WebSocket or SSE) |
| `POST /api/summary/{id}` | POST | Generate AI summary |
| `GET /api/report/{id}` | GET | Download HTML report |
| `GET /api/models` | GET | List available Ollama models |

---

## 14. ENVIRONMENT & HARDWARE

- **OS**: Windows 10/11
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **RAM**: 24 GB
- **Ollama**: v0.20.7
- **Python**: 3.10+
- **Models**: Primarily cloud models (zero local storage, remote execution)
