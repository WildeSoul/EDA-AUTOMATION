"""
Central configuration — all constants, model configs, timeouts, and system prompts.
"""
import os

# ──────────────────────────────────────────────
# Ollama Engine Settings
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_NUM_CTX = 8192          # Context window cap (fits 8 GB VRAM + 24 GB RAM)
OLLAMA_GPU_LAYERS = 99         # Max GPU offloading (RTX 4060 8 GB)

# ──────────────────────────────────────────────
# Model Ranking (sorted by preference)
# Higher = tried first in dropdowns & pipelines
# ──────────────────────────────────────────────
PREFERRED_ORDER = [
    # ── Tier 0: Flagship Cloud (strongest reasoning) ──
    "deepseek-v4-pro:cloud",        # ☁️ DeepSeek V4 Pro — best reasoning, agent orchestration
    "kimi-k2.6:cloud",              # ☁️ Kimi K2.6 — strong long-context, report writing
    # ── Tier 1: Frontier Cloud ──
    "gemma4:31b-cloud",             # ☁️ Google Gemma 4 — frontier reasoning + multimodal
    "qwen3.5:cloud",                # ☁️ Qwen 3.5 — strong data analysis & reasoning
    "kimi-k2.5:cloud",              # ☁️ Kimi K2.5 — multimodal agentic powerhouse
    "glm-5.1:cloud",                # ☁️ GLM 5.1 — agentic engineering, strong code
    # ── Tier 2: Specialist Cloud ──
    "qwen3-coder-next:cloud",       # ☁️ Qwen3 Coder — code generation specialist
    "minimax-m3:cloud",             # ☁️ MiniMax M3 — latest coding & productivity
    "nemotron-3-super:cloud",       # ☁️ NVIDIA — efficient MoE reasoning
    "nemotron-3-nano:30b-cloud",    # ☁️ NVIDIA — fast 30B executor
    "minimax-m2.7:cloud",           # ☁️ MiniMax M2.7 — coding & productivity
    # ── Tier 3: Local Models ──
    "gemma4:e4b",                   # 💻 9.6 GB — Gemma 4 extended local
    "gemma4:e2b",                   # 💻 7.2 GB — Gemma 4 efficient local
    "mistral-nemo:latest",          # 💻 7.1 GB — reliable local workhorse
    "llama3.1:latest",              # 💻 4.9 GB — lightweight backup
]

CLOUD_MODELS = {
    "deepseek-v4-pro:cloud",
    "kimi-k2.6:cloud",
    "gemma4:31b-cloud",
    "qwen3.5:cloud",
    "kimi-k2.5:cloud",
    "glm-5.1:cloud",
    "qwen3-coder-next:cloud",
    "minimax-m3:cloud",
    "nemotron-3-super:cloud",
    "nemotron-3-nano:30b-cloud",
    "minimax-m2.7:cloud",
}

# ──────────────────────────────────────────────
# Phase 1 Agent Model Assignments
# Dedicated models per role for consistency
# ──────────────────────────────────────────────
AGENT_MODELS = {
    "reasoning":  "deepseek-v4-pro:cloud",      # 🥇 Schema, Quality, EDA, Hypothesis agents
    "coding":     "qwen3-coder-next:cloud",      # 🥈 NL→SQL, Code Lab, Pipeline generation
    "reporting":  "kimi-k2.6:cloud",             # 🥉 Insight agent, Summaries, Reports
    "backup":     "gemma4:31b-cloud",            # 🔄 Fallback if primary unavailable
    "embedding":  "nomic-embed-text:latest",     # 🔤 Reserved for Phase 3 (ChromaDB)
}

# ──────────────────────────────────────────────
# Legacy Agentic Pipeline Config (Plan → Execute → Synthesize)
# Kept for backward compatibility with Streamlit UI
# ──────────────────────────────────────────────
NEMOTRON_MODELS = {
    "planner":        "deepseek-v4-pro:cloud",        # ☁️ Upgraded: DeepSeek V4 Pro
    "executor":       "qwen3.5:cloud",                # ☁️ Qwen 3.5 — precise data analysis executor
    "code_gen":       "qwen3-coder-next:cloud",       # ☁️ Qwen3 Coder — code generation specialist
    "executor_local": "mistral-nemo:latest",          # 💻 local fallback 12.2B
}

# ──────────────────────────────────────────────
# Agent Cache Configuration
# ──────────────────────────────────────────────
AGENT_CACHE_TTL = 3600             # Cache LLM results for 1 hour (seconds)

# ──────────────────────────────────────────────
# Memory Confidence Gate
# ──────────────────────────────────────────────
MEMORY_INJECTION_THRESHOLD = 0.65  # Threshold below which historical memory is ignored

# ──────────────────────────────────────────────
# SQLite Session Persistence
# ──────────────────────────────────────────────
SESSION_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions.db",
)

# ──────────────────────────────────────────────
# NVIDIA Cloud API (kept as fallback)
# ──────────────────────────────────────────────
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# ──────────────────────────────────────────────
# Timeouts (seconds)
# ──────────────────────────────────────────────
TIMEOUT_LOCAL = 300
TIMEOUT_CLOUD = 120
TIMEOUT_NVIDIA_API = 600

# ──────────────────────────────────────────────
# System Prompt — Elite Data Scientist
# ──────────────────────────────────────────────
SYSTEM_ROLE = (
    "You are an elite data scientist and statistician with 20+ years of experience in "
    "exploratory data analysis, statistical modelling, and business intelligence.\n\n"
    "CORE EXPERTISE you MUST apply:\n"
    "- DISTRIBUTIONS: Identify skewness, kurtosis, multi-modality; recommend transforms (log, Box-Cox, sqrt).\n"
    "- STATISTICAL TESTS: Apply and interpret t-tests, ANOVA, chi-square, Shapiro-Wilk, Mann-Whitney U. "
    "Report test statistics AND p-values.\n"
    "- CORRELATIONS: Distinguish correlation from causation. Use Pearson for linear, Spearman for monotonic. "
    "Flag spurious correlations.\n"
    "- FEATURE IMPORTANCE: Discuss which features drive variation. Suggest PCA loadings, mutual information, "
    "or domain-driven feature engineering.\n"
    "- OUTLIERS: Use IQR, Z-score, AND domain knowledge. Explain whether outliers are errors or real signals.\n"
    "- MISSING DATA: Assess MCAR/MAR/MNAR patterns. Recommend imputation strategies with trade-offs.\n\n"
    "STRICT RULES:\n"
    "1. CITE EXACT NUMBERS: column names, values, counts, percentages, p-values. NEVER say 'many' or 'some'.\n"
    "2. SHOW REASONING: Explain WHY a pattern exists, not just WHAT it is. Hypothesize causal mechanisms.\n"
    "3. CROSS-VALIDATE: Check findings across multiple dimensions. Flag if a pattern holds only for a subset.\n"
    "4. ACTIONABLE OUTPUT: Every recommendation needs specific thresholds, SQL/Python code, or steps.\n"
    "5. CONFIDENCE + CAVEATS: Rate High/Medium/Low confidence. State sample size, assumptions, limitations.\n"
    "6. BUSINESS IMPACT: Translate statistical findings into business language — revenue, risk, efficiency.\n"
    "7. THINK STEP-BY-STEP: Decompose complex questions before answering. Show your analytical chain."
)

# ──────────────────────────────────────────────
# Code Sandbox — allowed builtins and governance limits
# ──────────────────────────────────────────────
SANDBOX_TIMEOUT_SECONDS = 30
SANDBOX_MAX_MEMORY_MB = 512
SANDBOX_MAX_STDOUT_BYTES = 50 * 1024       # 50 KB output cap
SANDBOX_MAX_FIGURES = 10                   # Max 10 figures
SANDBOX_MAX_RESULT_ROWS = 5000             # Max 5,000 preview rows
SANDBOX_ALLOWED_MODULES = {
    "builtins", "math", "datetime", "json", "re", "collections",
    "itertools", "pandas", "numpy", "matplotlib", "seaborn",
    "scipy", "sklearn", "warnings", "functools", "operator", "copy", "types",
}

SANDBOX_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "format", "frozenset", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "print", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
}

# ──────────────────────────────────────────────
# ML Model Options
# ──────────────────────────────────────────────
ML_CLASSIFICATION_MODELS = ["Random Forest", "Logistic Regression", "XGBoost"]
ML_REGRESSION_MODELS = ["Random Forest", "Linear Regression", "XGBoost"]
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5

# ──────────────────────────────────────────────
# Anomaly Detection Defaults
# ──────────────────────────────────────────────
ANOMALY_CONTAMINATION_DEFAULT = 0.05
ANOMALY_ZSCORE_THRESHOLD = 3.0

# Load .env for API keys
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(os.path.dirname(_script_dir), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
