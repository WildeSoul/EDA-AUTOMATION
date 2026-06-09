import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests, json, textwrap
from itertools import combinations
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stMetricValue { font-size: 1.4rem !important; }
    .stMetricDelta { font-size: 0.9rem !important; }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Data Analyst Pro")
st.markdown(
    "Upload **any dataset** → get full **EDA + Advanced Analysis + Feature Engineering** → "
    "then **ask questions** and get detailed solutions, insights & advice from a local Ollama LLM."
)


# ──────────────────────────────────────────────
# Helper: talk to Ollama (local REST API)
# ──────────────────────────────────────────────
# Cap context window to prevent OOM — models like mistral-nemo default
# to 1M tokens which inflates RAM from ~7 GiB → 42+ GiB.
# 8192 gives good analytical depth while staying within 8 GB VRAM + 24 GB RAM.
OLLAMA_NUM_CTX = 8192

# Force maximum GPU offloading (RTX 4060 8 GB VRAM)
OLLAMA_GPU_LAYERS = 99  # Ollama will use as many as fit in VRAM

def query_ollama(model: str, prompt: str, system: str = "", timeout: int = 300) -> str:
    """Send a prompt to Ollama and return full response text."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_gpu": OLLAMA_GPU_LAYERS},
    }
    if system:
        payload["system"] = system
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.ConnectionError:
        return "❌ Could not connect to Ollama. Make sure `ollama serve` is running."
    except requests.HTTPError as e:
        # Detect out-of-memory errors from Ollama
        try:
            err_body = resp.json().get("error", "")
        except Exception:
            err_body = resp.text
        if "memory" in str(err_body).lower():
            return f"❌ Out of memory: {err_body}"
        return f"❌ Ollama HTTP error ({resp.status_code}): {err_body}"
    except Exception as e:
        return f"❌ Ollama error: {e}"


def stream_ollama(model: str, prompt: str, system: str = ""):
    """Stream tokens from the local Ollama server as a generator."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_gpu": OLLAMA_GPU_LAYERS},
    }
    if system:
        payload["system"] = system
    try:
        with requests.post(url, json=payload, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except requests.ConnectionError:
        yield "❌ Could not connect to Ollama. Make sure `ollama serve` is running."
    except Exception as e:
        yield f"❌ Ollama error: {e}"


def unload_ollama_model(model: str):
    """Tell Ollama to unload a model from memory to free RAM/VRAM."""
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "keep_alive": 0, "options": {"num_ctx": OLLAMA_NUM_CTX}},
            timeout=10,
        )
    except Exception:
        pass  # Best effort — ignore failures


# ──────────────────────────────────────────────
# Model ranking — updated to match installed models
# ──────────────────────────────────────────────
PREFERRED_ORDER = [
    "nemotron-3-super:cloud",   # ☁️ Cloud — NVIDIA Nemotron Super (best reasoning/planning)
    "nemotron-3-nano:30b-cloud",# ☁️ Cloud — NVIDIA Nemotron Nano 30B (fast executor)
    "minimax-m2.7:cloud",       # ☁️ Cloud — MiniMax (high quality)
    "mistral-nemo:latest",      # 💻 12.2B — reliable local workhorse
    "llama3.1:latest",          # 💻  8B  — lightweight backup
]

CLOUD_MODELS = {"minimax-m2.7:cloud", "nemotron-3-super:cloud", "nemotron-3-nano:30b-cloud"}  # models that run via cloud API

# ──────────────────────────────────────────────
# Nemotron Super+Nano Agentic Config
# ──────────────────────────────────────────────
NEMOTRON_MODELS = {
    "planner": "nemotron-3-super:cloud",       # ☁️ Ollama cloud — heavy reasoning & planning
    "executor": "nemotron-3-nano:30b-cloud",   # ☁️ Ollama cloud — fast 30B executor
    "executor_local": "mistral-nemo:latest",   # 💻 Local fallback — 12.2B
}
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# Load .env file for API keys (use script directory so it works regardless of CWD)
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _env_path = os.path.join(_script_dir, ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed, use sidebar input only

_ENV_NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")

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

# Timeout settings per model type
TIMEOUT_LOCAL = 300
TIMEOUT_CLOUD = 120
TIMEOUT_NVIDIA_API = 600  # Super gets generous timeout for complex reasoning


def get_ollama_models() -> list[str]:
    """Fetch locally installed Ollama models, sorted by suitability."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            return ["(no models found)"]

        def sort_key(name):
            try:
                return PREFERRED_ORDER.index(name)
            except ValueError:
                return len(PREFERRED_ORDER)
        return sorted(models, key=sort_key)
    except Exception:
        return ["(ollama not reachable)"]


# ──────────────────────────────────────────────
# NVIDIA Cloud API — for Nemotron Super (Planner)
# ──────────────────────────────────────────────
def query_nvidia_api(prompt: str, system: str = "", api_key: str = "",
                     model: str = "", timeout: int = TIMEOUT_NVIDIA_API) -> str:
    """Call NVIDIA's integrate API (OpenAI-compatible) for Nemotron Super."""
    if not api_key:
        return "❌ NVIDIA API key not set. Enter it in the sidebar."
    url = f"{NVIDIA_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model or NEMOTRON_MODELS["planner"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # Nemotron Super may return reasoning in 'reasoning_content' and answer in 'content'
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        # If content is empty/null but reasoning has data, use reasoning as fallback
        if not content.strip() and reasoning.strip():
            return reasoning.strip()
        return content
    except requests.ConnectionError:
        return "❌ Could not connect to NVIDIA API. Check your internet connection."
    except requests.HTTPError as e:
        if resp.status_code == 401:
            return "❌ Invalid NVIDIA API key. Check your key at build.nvidia.com."
        return f"❌ NVIDIA API error ({resp.status_code}): {e}"
    except Exception as e:
        return f"❌ NVIDIA API error: {e}"


# ──────────────────────────────────────────────
# Agentic Pipeline: Plan → Execute → Synthesize
# ──────────────────────────────────────────────
def agentic_plan(question: str, data_context: str, api_key: str) -> tuple[str, list[str]]:
    """
    STEP 1 — Super decomposes the question into analytical sub-tasks.
    Returns (raw_plan_text, list_of_steps).
    """
    plan_prompt = textwrap.dedent(f"""\
    You are a principal data scientist designing an analysis plan. Decompose this
    question into exactly 3-5 numbered sub-tasks that cover the FULL analytical depth.

    Each sub-task MUST:
    - NAME specific columns, statistical methods, and expected output format
    - Be INDEPENDENTLY executable with the provided data
    - Include one of: descriptive stats, distribution analysis, correlation check,
      hypothesis test, segmentation, trend analysis, or anomaly detection
    - State the STATISTICAL METHOD (e.g., "compute Pearson r between X and Y",
      "test normality of Z with Shapiro-Wilk", "segment by Q and compare medians")

    ENSURE coverage of:
    - At least 1 sub-task on data quality / distribution shape
    - At least 1 sub-task on relationships / correlations
    - At least 1 sub-task on actionable segmentation or comparison

    ## USER QUESTION
    {question}

    ## AVAILABLE DATA CONTEXT
    {data_context[:4000]}

    ## OUTPUT FORMAT
    Return ONLY the numbered list, one sub-task per line:
    1. [sub-task with specific columns and statistical method]
    2. [sub-task with specific columns and statistical method]
    ...
    """)

    plan_system = (
        "You are a principal data scientist who designs rigorous analysis plans. "
        "Every sub-task must name specific columns, statistical tests, and metrics. "
        "You ensure completeness: data quality, distributions, relationships, and actionable insights."
    )

    planner_model = NEMOTRON_MODELS["planner"]
    raw_plan = query_ollama(planner_model, plan_prompt, system=plan_system,
                            timeout=TIMEOUT_CLOUD)

    # Parse numbered steps
    steps = []
    for line in raw_plan.split("\n"):
        line = line.strip()
        if line and len(line) > 3 and line[0].isdigit() and ("." in line[:3] or ")" in line[:3]):
            # Remove the numbering prefix
            step_text = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if step_text:
                steps.append(step_text)

    if not steps:
        # Fallback: treat the whole response as a single step
        steps = [question]

    return raw_plan, steps


def agentic_execute(steps: list[str], data_context: str) -> list[dict]:
    """
    STEP 2 — Nano executes each sub-task against the data context.
    Returns list of {step, result} dicts.
    """
    executor_model = NEMOTRON_MODELS["executor"]
    executor_is_cloud = executor_model in CLOUD_MODELS
    exec_system = (
        "You are a precise quantitative analyst executing a specific analytical task. "
        "RULES:\n"
        "1. Use ONLY the data provided — never fabricate numbers.\n"
        "2. Cite exact column names, values, counts, percentages, p-values.\n"
        "3. For statistical tests, report: test_statistic, p-value, effect_size, interpretation.\n"
        "4. For distributions, report: mean, median, std, skewness, range.\n"
        "5. For correlations, report: coefficient, direction, strength (weak/moderate/strong).\n"
        "6. Keep response focused — answer ONLY this specific sub-task.\n"
        "7. End with one sentence on confidence level and any data quality caveats."
    )

    results = []
    for step in steps:
        exec_prompt = textwrap.dedent(f"""\
        ## TASK
        {step}

        ## DATA
        {data_context[:4000]}

        Answer with EXACT numbers from the data above. Structure your response:
        - **Finding**: The specific answer with numbers
        - **Evidence**: Supporting statistics (means, medians, p-values, counts)
        - **Caveat**: Any data quality issue affecting this finding
        """)

        exec_timeout = TIMEOUT_CLOUD if executor_is_cloud else TIMEOUT_LOCAL
        result = query_ollama(executor_model, exec_prompt, system=exec_system,
                              timeout=exec_timeout)
        results.append({"step": step, "result": result})

    return results


def agentic_synthesize(question: str, plan: str, exec_results: list[dict],
                       api_key: str) -> str:
    """
    STEP 3 — Super merges the plan + execution results into a polished answer.
    """
    results_text = ""
    for i, r in enumerate(exec_results, 1):
        results_text += f"\n### Sub-task {i}: {r['step']}\n{r['result']}\n"

    synth_prompt = textwrap.dedent(f"""\
    You are a principal data scientist creating an executive-level synthesis.

    ## ORIGINAL QUESTION
    {question}

    ## ANALYSIS PLAN
    {plan}

    ## EXECUTION RESULTS FROM ANALYST
    {results_text}

    ## YOUR TASK
    Synthesize ALL sub-task results into a rigorous, comprehensive answer.

    ### 📌 DIRECT ANSWER
    Lead with the single most important finding. Include the key number.

    ### 📊 STATISTICAL EVIDENCE
    Summarize the hard numbers: means, medians, correlations, p-values,
    effect sizes from the execution results. Use a table if comparing groups.

    ### 🔍 KEY INSIGHTS (3-5)
    Numbered insights — each must cite a SPECIFIC number from the execution results.
    For each insight, explain the MECHANISM (why this pattern exists).

    ### ⚠️ DATA QUALITY & CAVEATS
    Flag sample size issues, missing data impact, violated assumptions,
    or any reason the findings might not generalize.

    ### ✅ ACTIONABLE RECOMMENDATIONS
    Numbered steps with specific thresholds, filters, or code snippets.
    Each must be directly tied to an insight above.

    ### 🎯 CONFIDENCE LEVEL
    Rate High/Medium/Low.
    - State key ASSUMPTIONS made
    - State what EVIDENCE would change the conclusion
    - State recommended SAMPLE SIZE for stronger conclusions

    CRITICAL: Use EXACT numbers from execution results. NEVER fabricate data.
    Cross-check that insights are consistent across sub-tasks.
    """)

    synth_system = (
        "You are a principal data scientist creating executive-level synthesis of "
        "analytical findings. You combine statistical rigor with business clarity. "
        "Cite exact numbers, quantify uncertainty, and provide specific actionable steps."
    )

    planner_model = NEMOTRON_MODELS["planner"]
    return query_ollama(planner_model, synth_prompt, system=synth_system,
                        timeout=TIMEOUT_CLOUD)


# ──────────────────────────────────────────────
# Feature Engineering Functions
# ──────────────────────────────────────────────
def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect columns that look like dates."""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True)
                if parsed.notna().sum() > len(sample) * 0.7:
                    date_cols.append(col)
            except (ValueError, TypeError):
                continue
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
    return date_cols


def engineer_date_features(df: pd.DataFrame, date_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Extract temporal features from date columns."""
    new_features = []
    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        prefix = col.replace(" ", "_")
        df[f"{prefix}_year"] = dt.dt.year
        df[f"{prefix}_month"] = dt.dt.month
        df[f"{prefix}_day"] = dt.dt.day
        df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
        df[f"{prefix}_quarter"] = dt.dt.quarter
        df[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        new_features.extend([
            f"{prefix}_year", f"{prefix}_month", f"{prefix}_day",
            f"{prefix}_dayofweek", f"{prefix}_quarter", f"{prefix}_is_weekend"
        ])
    return df, new_features


def engineer_interaction_features(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Create interaction features from top correlated numeric pairs."""
    new_features = []
    if len(numeric_cols) < 2:
        return df, new_features

    corr_matrix = df[numeric_cols].corr().abs()
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for c1, c2, _ in pairs[:top_n]:
        feat_name = f"{c1}_x_{c2}"
        df[feat_name] = df[c1] * df[c2]
        new_features.append(feat_name)

        if (df[c2] != 0).all():
            ratio_name = f"{c1}_div_{c2}"
            df[ratio_name] = df[c1] / df[c2].replace(0, np.nan)
            new_features.append(ratio_name)

    return df, new_features


def compute_skewness(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Identify skewed features and suggest transforms."""
    skew_data = []
    for col in numeric_cols:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            direction = "right-skewed" if skew_val > 0 else "left-skewed"
            suggestion = "log transform" if skew_val > 1 else "square transform"
            skew_data.append({
                "Column": col,
                "Skewness": round(skew_val, 3),
                "Direction": direction,
                "Suggested Fix": suggestion
            })
    return pd.DataFrame(skew_data) if skew_data else pd.DataFrame()


def impute_missing(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Impute missing values — median for numeric, mode for categorical."""
    report = {}
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report[col] = f"filled {n_missing} NaN → median ({median_val:.2f})"

    for col in categorical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            report[col] = f"filled {n_missing} NaN → mode ('{mode_val}')"

    return df, report


# ──────────────────────────────────────────────
# Advanced Statistical Analysis Functions
# ──────────────────────────────────────────────
def run_normality_tests(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Run Shapiro-Wilk and D'Agostino normality tests on numeric columns."""
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 8:
            continue
        # Use sample if too large for Shapiro-Wilk (max ~5000)
        sample = data.sample(min(5000, len(data)), random_state=42)
        try:
            sw_stat, sw_p = scipy_stats.shapiro(sample)
        except Exception:
            sw_stat, sw_p = np.nan, np.nan
        try:
            da_stat, da_p = scipy_stats.normaltest(sample)
        except Exception:
            da_stat, da_p = np.nan, np.nan

        is_normal = (sw_p > 0.05) and (da_p > 0.05) if not (np.isnan(sw_p) or np.isnan(da_p)) else None
        results.append({
            "Column": col,
            "Shapiro-Wilk stat": round(sw_stat, 4) if not np.isnan(sw_stat) else "N/A",
            "Shapiro p-value": round(sw_p, 4) if not np.isnan(sw_p) else "N/A",
            "D'Agostino stat": round(da_stat, 4) if not np.isnan(da_stat) else "N/A",
            "D'Agostino p-value": round(da_p, 4) if not np.isnan(da_p) else "N/A",
            "Normal?": "✅ Yes" if is_normal else ("❌ No" if is_normal is not None else "⚠️ N/A"),
        })
    return pd.DataFrame(results) if results else pd.DataFrame()


def run_chi_square_tests(df: pd.DataFrame, categorical_cols: list[str], max_pairs: int = 10) -> pd.DataFrame:
    """Run chi-square independence tests between pairs of categorical columns."""
    results = []
    pairs_tested = 0
    for i, c1 in enumerate(categorical_cols):
        if pairs_tested >= max_pairs:
            break
        for c2 in categorical_cols[i+1:]:
            if pairs_tested >= max_pairs:
                break
            # Skip columns with too many unique values
            if df[c1].nunique() > 20 or df[c2].nunique() > 20:
                continue
            try:
                contingency = pd.crosstab(df[c1], df[c2])
                chi2, p_val, dof, _ = scipy_stats.chi2_contingency(contingency)
                cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1))) if min(contingency.shape) > 1 else 0
                results.append({
                    "Variable 1": c1,
                    "Variable 2": c2,
                    "Chi² Statistic": round(chi2, 2),
                    "p-value": round(p_val, 4),
                    "Cramér's V": round(cramers_v, 3),
                    "Significant?": "✅ Yes" if p_val < 0.05 else "❌ No",
                    "Strength": "🔴 Strong" if cramers_v > 0.5 else ("🟡 Moderate" if cramers_v > 0.3 else "🔵 Weak"),
                })
                pairs_tested += 1
            except Exception:
                continue
    return pd.DataFrame(results) if results else pd.DataFrame()


def run_pca_analysis(df: pd.DataFrame, numeric_cols: list[str], n_components: int = 0):
    """Run PCA on standardized numeric columns. Returns pca, transformed data, scaler."""
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 3 or len(numeric_cols) < 2:
        return None, None, None, None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)

    n_comp = min(n_components if n_components > 0 else len(numeric_cols), len(numeric_cols), len(clean_df))
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(scaled)
    return pca, transformed, clean_df.index, numeric_cols


def run_kmeans_clustering(transformed_data, max_k: int = 8):
    """Find optimal k using elbow method and run KMeans."""
    if transformed_data is None or len(transformed_data) < 4:
        return None, None, None

    max_k = min(max_k, len(transformed_data) - 1)
    if max_k < 2:
        return None, None, None

    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(transformed_data[:, :min(5, transformed_data.shape[1])])
        inertias.append(km.inertia_)

    # Simple elbow detection: biggest drop in inertia rate
    if len(inertias) >= 2:
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        ratios = [diffs[i] / diffs[i+1] if diffs[i+1] > 0 else 1 for i in range(len(diffs)-1)]
        best_k = list(k_range)[np.argmax(ratios) + 1] if ratios else 2
    else:
        best_k = 2

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(transformed_data[:, :min(5, transformed_data.shape[1])])
    return labels, best_k, list(zip(k_range, inertias))


# ──────────────────────────────────────────────
# Build rich data context for LLM
# ──────────────────────────────────────────────
def build_data_context(df, uploaded_name, numeric_cols, categorical_cols,
                       engineered_features=None, imputation_report=None,
                       skew_df=None, date_cols=None, user_question=""):
    """Build a comprehensive data context string for the LLM."""
    ctx = []
    ctx.append(f"DATASET: {uploaded_name}")
    ctx.append(f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    ctx.append(f"COLUMNS & TYPES: {dict(zip(df.columns, df.dtypes.astype(str)))}")

    # Date range info
    if date_cols:
        for dc in date_cols:
            dt_series = pd.to_datetime(df[dc], errors="coerce")
            min_d, max_d = dt_series.min(), dt_series.max()
            ctx.append(f"DATE RANGE ({dc}): {min_d} to {max_d}")

    # Descriptive statistics
    if numeric_cols:
        ctx.append(f"\n--- NUMERICAL STATISTICS ---\n{df[numeric_cols].describe().to_string()}")

    # Categorical value counts
    if categorical_cols:
        ctx.append("\n--- CATEGORICAL BREAKDOWN ---")
        for c in categorical_cols[:8]:
            vc = df[c].value_counts().head(7).to_dict()
            ctx.append(f"  '{c}': {vc}")

    # Top correlations
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Var1", "Var2", "Correlation"]
        top_pos = pairs.nlargest(5, "Correlation")
        top_neg = pairs.nsmallest(5, "Correlation")
        ctx.append("\n--- TOP POSITIVE CORRELATIONS ---")
        for _, r in top_pos.iterrows():
            ctx.append(f"  {r.Var1} ↔ {r.Var2}: {r.Correlation:.3f}")
        ctx.append("\n--- TOP NEGATIVE CORRELATIONS ---")
        for _, r in top_neg.iterrows():
            ctx.append(f"  {r.Var1} ↔ {r.Var2}: {r.Correlation:.3f}")

    # Outlier counts
    if numeric_cols:
        ctx.append("\n--- OUTLIER COUNTS (IQR method) ---")
        for c in numeric_cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out_count = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
                if out_count > 0:
                    ctx.append(f"  {c}: {out_count} outliers ({out_count/len(df)*100:.1f}%)")

    # Missing & duplicates
    ctx.append(f"\nMISSING VALUES: {int(df.isnull().sum().sum())} total")
    per_col_missing = df.isnull().sum()
    per_col_missing = per_col_missing[per_col_missing > 0]
    if not per_col_missing.empty:
        ctx.append(f"  Per column: {per_col_missing.to_dict()}")
    ctx.append(f"DUPLICATE ROWS: {int(df.duplicated().sum())}")

    # Engineered features
    if engineered_features:
        ctx.append(f"\n--- ENGINEERED FEATURES ({len(engineered_features)} new) ---")
        ctx.append(f"  {', '.join(engineered_features)}")

    # Imputation report
    if imputation_report:
        ctx.append("\n--- DATA IMPUTATION PERFORMED ---")
        for col, action in imputation_report.items():
            ctx.append(f"  {col}: {action}")

    # Skewness
    if skew_df is not None and not skew_df.empty:
        ctx.append("\n--- SKEWED FEATURES ---")
        for _, row in skew_df.iterrows():
            ctx.append(f"  {row['Column']}: skewness={row['Skewness']} ({row['Direction']}) → {row['Suggested Fix']}")

    # Relevant rows for the question
    if user_question:
        numbers_in_q = re.findall(r"[\d,]+\.?\d*", user_question.replace(",", ""))
        numbers_in_q = [float(n) for n in numbers_in_q if n]
        keywords = [w for w in user_question.lower().split() if len(w) > 2]

        relevant_rows = pd.DataFrame()
        # Try numeric matching
        if numbers_in_q and numeric_cols:
            target_val = max(numbers_in_q)
            for nc in numeric_cols:
                mask = (df[nc] >= target_val * 0.8) & (df[nc] <= target_val * 1.2)
                matched = df[mask]
                if len(matched) > 0:
                    relevant_rows = matched.head(10)
                    ctx.append(f"\n--- ROWS MATCHING ~{target_val:,.0f} (±20%) in '{nc}' ({len(matched)} total) ---")
                    ctx.append(relevant_rows.to_string(index=False))
                    break

        # Try keyword matching in categorical columns
        if relevant_rows.empty and keywords and categorical_cols:
            for cat_col in categorical_cols:
                str_vals = df[cat_col].astype(str).str.lower()
                for kw in keywords:
                    mask = str_vals.str.contains(kw, na=False)
                    if mask.sum() > 0:
                        relevant_rows = df[mask].head(10)
                        ctx.append(f"\n--- ROWS MATCHING keyword '{kw}' in '{cat_col}' ({mask.sum()} total) ---")
                        ctx.append(relevant_rows.to_string(index=False))
                        break
                if not relevant_rows.empty:
                    break

        # Fallback to random sample
        if relevant_rows.empty:
            sample = df.sample(min(10, len(df)), random_state=42)
            ctx.append(f"\n--- SAMPLE ROWS (random 10) ---\n{sample.to_string(index=False)}")

    return "\n".join(ctx)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("1️⃣  Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )

    st.header("2️⃣  AI Model (Local Ollama)")
    available_models = get_ollama_models()
    # Format model names with cloud/local badges
    model_labels = {
        m: f"☁️ {m}" if m in CLOUD_MODELS else f"💻 {m}"
        for m in available_models
    }
    selected_model = st.selectbox(
        "Select Ollama Model", available_models,
        format_func=lambda m: model_labels.get(m, m),
    )
    if selected_model in CLOUD_MODELS:
        st.caption("☁️ Cloud model — runs via MiniMax API through Ollama.")
    else:
        st.caption("💻 Local model — runs entirely on your machine.")

    st.header("3️⃣  Options")
    run_feature_eng = st.checkbox("🔧 Auto Feature Engineering", value=True)
    run_advanced_stats = st.checkbox("📐 Advanced Statistical Tests", value=True,
                                     help="Normality tests, Chi-square, PCA & Clustering")
    run_auto_summary = st.checkbox("🤖 AI Auto-Summary", value=False,
                                    help="Generate an executive AI summary on load")
    run_consensus = st.checkbox("🔀 Multi-Model Consensus", value=False,
                                help="Query multiple models and merge answers (Q&A)")


# ──────────────────────────────────────────────
# Load data with caching
# ──────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes, file_name):
    if file_name.endswith(".csv"):
        return pd.read_csv(file_bytes)
    else:
        return pd.read_excel(file_bytes)


# ──────────────────────────────────────────────
# Main body — only when a file is uploaded
# ──────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file, uploaded_file.name)
        df_original = df.copy()  # keep original for reference

        st.success(f"✅ Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

        # ============================================================
        # SECTION 1 — Data Inspection & Health Check
        # ============================================================
        st.header("🔍 1. Data Inspection & Health Check")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicates", int(df.duplicated().sum()))
        col4.metric("Missing Cells", int(df.isnull().sum().sum()))
        col5.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")

        with st.expander("📋 Data Preview (first 20 rows)", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Column Data Types")
            dtypes_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("Missing Values per Column")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if missing.empty:
                st.write("No missing values found! 🎉")
            else:
                missing_df = missing.reset_index()
                missing_df.columns = ["Column", "Missing Count"]
                missing_df["% Missing"] = (
                    (missing_df["Missing Count"] / len(df) * 100).round(2)
                )
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

                # Visualize missing values
                fig, ax = plt.subplots(figsize=(10, max(3, len(missing) * 0.4)))
                colors = ["#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71"
                          for v in missing_df["% Missing"]]
                ax.barh(missing_df["Column"], missing_df["% Missing"], color=colors)
                ax.set_xlabel("% Missing")
                ax.set_title("Missing Values Distribution")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ============================================================
        # SECTION 2 — Statistical Profiling
        # ============================================================
        st.header("📈 2. Statistical Profiling")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        tab_num, tab_cat, tab_full = st.tabs(["Numerical Summary", "Categorical Summary", "Full Describe"])
        with tab_num:
            if numeric_cols:
                desc = df[numeric_cols].describe().T
                desc["skewness"] = df[numeric_cols].skew()
                desc["kurtosis"] = df[numeric_cols].kurtosis()
                st.dataframe(desc.round(3), use_container_width=True)
            else:
                st.info("No numerical columns detected.")
        with tab_cat:
            if categorical_cols:
                cat_summary = []
                for col in categorical_cols:
                    cat_summary.append({
                        "Column": col,
                        "Unique Values": df[col].nunique(),
                        "Top Value": df[col].mode()[0] if not df[col].mode().empty else "N/A",
                        "Top Frequency": df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                        "Missing": df[col].isnull().sum(),
                    })
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True)
            else:
                st.info("No categorical columns detected.")
        with tab_full:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        # ============================================================
        # SECTION 3 — Automated Feature Engineering
        # ============================================================
        engineered_features = []
        imputation_report = {}
        skew_df = pd.DataFrame()
        date_cols = []

        if run_feature_eng:
            st.header("🔧 3. Automated Feature Engineering")

            fe_steps = []

            # 3a — Date feature extraction
            date_cols = detect_date_columns(df)
            if date_cols:
                df, date_feats = engineer_date_features(df, date_cols)
                engineered_features.extend(date_feats)
                fe_steps.append(f"📅 **Date features**: Extracted {len(date_feats)} temporal features from `{'`, `'.join(date_cols)}`")

            # 3b — Missing value imputation
            # Re-detect after date features
            numeric_cols_for_impute = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols_for_impute = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if df.isnull().sum().sum() > 0:
                df, imputation_report = impute_missing(df, numeric_cols_for_impute, cat_cols_for_impute)
                fe_steps.append(f"🩹 **Imputation**: Fixed {len(imputation_report)} columns with missing values")

            # 3c — Skewness detection
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            skew_df = compute_skewness(df, numeric_cols)
            if not skew_df.empty:
                fe_steps.append(f"📐 **Skewness detected**: {len(skew_df)} features are highly skewed")

            # 3d — Interaction features
            original_numeric = [c for c in numeric_cols if c not in engineered_features]
            if len(original_numeric) >= 2:
                df, interaction_feats = engineer_interaction_features(df, original_numeric, top_n=3)
                engineered_features.extend(interaction_feats)
                fe_steps.append(f"✖️ **Interaction features**: Created {len(interaction_feats)} features (products & ratios)")

            # Update column lists after engineering
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Display summary
            if fe_steps:
                for step in fe_steps:
                    st.markdown(f"- {step}")
            else:
                st.info("No feature engineering steps were applicable to this dataset.")

            # Show details in expandable sections
            if imputation_report:
                with st.expander("🩹 Imputation Details"):
                    imp_df = pd.DataFrame([
                        {"Column": k, "Action": v} for k, v in imputation_report.items()
                    ])
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)

            if not skew_df.empty:
                with st.expander("📐 Skewness Details & Suggestions"):
                    st.dataframe(skew_df, use_container_width=True, hide_index=True)

            if engineered_features:
                with st.expander(f"🆕 Engineered Features Preview ({len(engineered_features)} new columns)"):
                    preview_cols = [c for c in engineered_features if c in df.columns]
                    st.dataframe(df[preview_cols].head(10), use_container_width=True)

        section_offset = 1 if run_feature_eng else 0

        # ============================================================
        # SECTION 4 — Automated Visualizations
        # ============================================================
        st.header(f"🎨 {3 + section_offset}. Automated Visualizations")

        viz_tabs = st.tabs(["📊 Distributions", "📦 Categories", "🔥 Correlations",
                           "📈 Pairwise", "📉 Time Series"])

        # --- Distributions ---
        with viz_tabs[0]:
            if numeric_cols:
                selected_num = st.selectbox("Pick a numerical column:", numeric_cols,
                                           key="dist_num")
                fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                # Histogram + KDE
                sns.histplot(df[selected_num].dropna(), kde=True, ax=axes[0], color="#3498db")
                axes[0].set_title(f"Distribution — {selected_num}")
                # Box Plot
                sns.boxplot(x=df[selected_num].dropna(), ax=axes[1], color="#2ecc71")
                axes[1].set_title(f"Box Plot — {selected_num}")
                # Violin Plot
                sns.violinplot(x=df[selected_num].dropna(), ax=axes[2], color="#9b59b6")
                axes[2].set_title(f"Violin — {selected_num}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No numerical columns to visualize.")

        # --- Categories ---
        with viz_tabs[1]:
            if categorical_cols:
                selected_cat = st.selectbox("Pick a categorical column:", categorical_cols,
                                           key="cat_col")
                top = df[selected_cat].value_counts().nlargest(15)
                fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
                bars = ax.barh(range(len(top)), top.values, color=sns.color_palette("viridis", len(top)))
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(top.index)
                ax.set_title(f"Top Categories — {selected_cat}")
                ax.set_xlabel("Count")
                ax.invert_yaxis()
                # Add value labels
                for bar, val in zip(bars, top.values):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                            f'{val:,}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No categorical columns detected.")

        # --- Correlations ---
        with viz_tabs[2]:
            if len(numeric_cols) > 1:
                # Limit to 15 columns for readability
                corr_cols = numeric_cols[:15]
                fig, ax = plt.subplots(figsize=(min(14, len(corr_cols) * 1.2),
                                                min(10, len(corr_cols) * 0.9)))
                corr = df[corr_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, cmap="RdBu_r", fmt=".2f",
                           ax=ax, linewidths=0.5, center=0, vmin=-1, vmax=1,
                           square=True)
                ax.set_title("Correlation Matrix (lower triangle)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Top correlations table
                with st.expander("📊 Top Correlations Table"):
                    pairs = (
                        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        .stack()
                        .reset_index()
                    )
                    pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
                    pairs["Abs Corr"] = pairs["Correlation"].abs()
                    pairs = pairs.sort_values("Abs Corr", ascending=False).head(15)
                    st.dataframe(pairs.drop(columns=["Abs Corr"]).round(3),
                               use_container_width=True, hide_index=True)
            else:
                st.info("Need at least 2 numerical columns for correlation analysis.")

        # --- Pairwise ---
        with viz_tabs[3]:
            if 2 <= len(numeric_cols) <= 10:
                pair_cols = numeric_cols[:6]  # limit for performance
                st.caption(f"Showing pairplot for: {', '.join(pair_cols)}")
                hue_col = None
                if categorical_cols:
                    if df[categorical_cols[0]].nunique() <= 6:
                        hue_col = categorical_cols[0]

                fig = sns.pairplot(df[pair_cols + ([hue_col] if hue_col else [])].dropna(),
                                  hue=hue_col, diag_kind="kde",
                                  palette="Set2" if hue_col else None,
                                  plot_kws={"alpha": 0.6, "s": 20})
                st.pyplot(fig.figure)
                plt.close(fig.figure)
            elif len(numeric_cols) > 10:
                st.info("Too many numeric columns for pairplot. Select up to 6 columns manually.")
                pair_selected = st.multiselect("Select columns for pairplot:",
                                               numeric_cols, default=numeric_cols[:4],
                                               key="pair_select")
                if 2 <= len(pair_selected) <= 6:
                    fig = sns.pairplot(df[pair_selected].dropna(), diag_kind="kde",
                                      plot_kws={"alpha": 0.6, "s": 20})
                    st.pyplot(fig.figure)
                    plt.close(fig.figure)
            else:
                st.info("Need at least 2 numerical columns for pairplot.")

        # --- Time Series ---
        with viz_tabs[4]:
            if date_cols:
                ts_date_col = st.selectbox("Date column:", date_cols, key="ts_date")
                ts_val_col = st.selectbox("Value column:", numeric_cols, key="ts_val")

                ts_df = df[[ts_date_col, ts_val_col]].copy()
                ts_df[ts_date_col] = pd.to_datetime(ts_df[ts_date_col], errors="coerce")
                ts_df = ts_df.dropna().sort_values(ts_date_col)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(ts_df[ts_date_col], ts_df[ts_val_col], color="#3498db", alpha=0.7, linewidth=1)
                # Rolling average
                if len(ts_df) > 10:
                    window = max(7, len(ts_df) // 30)
                    rolling_avg = ts_df[ts_val_col].rolling(window=window, center=True).mean()
                    ax.plot(ts_df[ts_date_col], rolling_avg, color="#e74c3c",
                            linewidth=2, label=f"{window}-point moving avg")
                    ax.legend()
                ax.set_title(f"{ts_val_col} over time")
                ax.set_xlabel(ts_date_col)
                ax.set_ylabel(ts_val_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No date columns detected for time series analysis.")

        # ============================================================
        # SECTION — Advanced Statistical Tests (NEW)
        # ============================================================
        if run_advanced_stats:
            st.header(f"📐 {4 + section_offset}. Advanced Statistical Tests")
            section_offset += 1

            stat_tabs = st.tabs(["🔔 Normality Tests", "🔗 Chi-Square", "📉 PCA", "🎯 Clustering"])

            with stat_tabs[0]:
                st.subheader("Distribution Normality Tests")
                st.caption("Tests whether each numeric column follows a normal distribution (p > 0.05 = normal).")
                if numeric_cols:
                    with st.spinner("Running normality tests..."):
                        norm_df = run_normality_tests(df, numeric_cols)
                    if not norm_df.empty:
                        normal_count = (norm_df["Normal?"] == "✅ Yes").sum()
                        non_normal_count = (norm_df["Normal?"] == "❌ No").sum()
                        cn1, cn2 = st.columns(2)
                        cn1.metric("Normal Distributions", normal_count)
                        cn2.metric("Non-Normal Distributions", non_normal_count)
                        st.dataframe(norm_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Not enough data for normality tests.")
                else:
                    st.info("No numeric columns available.")

            with stat_tabs[1]:
                st.subheader("Chi-Square Independence Tests")
                st.caption("Tests whether pairs of categorical variables are statistically independent.")
                if len(categorical_cols) >= 2:
                    with st.spinner("Running chi-square tests..."):
                        chi_df = run_chi_square_tests(df, categorical_cols)
                    if not chi_df.empty:
                        st.dataframe(chi_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No suitable categorical column pairs found.")
                else:
                    st.info("Need at least 2 categorical columns for chi-square tests.")

            with stat_tabs[2]:
                st.subheader("Principal Component Analysis (PCA)")
                if len(numeric_cols) >= 2:
                    with st.spinner("Running PCA..."):
                        pca_model, pca_data, pca_idx, pca_cols = run_pca_analysis(df, numeric_cols)
                    if pca_model is not None:
                        # Explained variance chart
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        cumulative = np.cumsum(pca_model.explained_variance_ratio_)
                        n_pc = len(pca_model.explained_variance_ratio_)
                        axes[0].bar(range(1, n_pc + 1), pca_model.explained_variance_ratio_,
                                   color="#3498db", alpha=0.7, label="Individual")
                        axes[0].plot(range(1, n_pc + 1), cumulative, "ro-", label="Cumulative")
                        axes[0].axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="95% threshold")
                        axes[0].set_xlabel("Principal Component")
                        axes[0].set_ylabel("Explained Variance Ratio")
                        axes[0].set_title("PCA Explained Variance")
                        axes[0].legend()

                        # 2D scatter
                        if pca_data.shape[1] >= 2:
                            scatter = axes[1].scatter(pca_data[:, 0], pca_data[:, 1],
                                                     alpha=0.5, s=15, c="#3498db")
                            axes[1].set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
                            axes[1].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
                            axes[1].set_title("PCA 2D Projection")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Components table
                        n_show = min(5, n_pc)
                        with st.expander(f"Top {n_show} Component Loadings"):
                            loadings = pd.DataFrame(
                                pca_model.components_[:n_show],
                                columns=pca_cols,
                                index=[f"PC{i+1}" for i in range(n_show)]
                            ).round(3)
                            st.dataframe(loadings, use_container_width=True)
                    else:
                        st.info("Not enough data for PCA.")
                else:
                    st.info("Need at least 2 numeric columns for PCA.")

            with stat_tabs[3]:
                st.subheader("Auto-Clustering (KMeans)")
                if len(numeric_cols) >= 2 and pca_data is not None:
                    with st.spinner("Finding optimal clusters..."):
                        cluster_labels, best_k, elbow_data = run_kmeans_clustering(pca_data)
                    if cluster_labels is not None:
                        st.success(f"Optimal number of clusters: **{best_k}** (detected via elbow method)")

                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        # Elbow chart
                        ks, inertias = zip(*elbow_data)
                        axes[0].plot(ks, inertias, "bo-")
                        axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"Optimal k={best_k}")
                        axes[0].set_xlabel("Number of Clusters (k)")
                        axes[0].set_ylabel("Inertia")
                        axes[0].set_title("Elbow Method")
                        axes[0].legend()

                        # Cluster scatter on PCA
                        if pca_data.shape[1] >= 2:
                            scatter = axes[1].scatter(pca_data[:, 0], pca_data[:, 1],
                                                     c=cluster_labels, cmap="Set2",
                                                     alpha=0.6, s=15)
                            axes[1].set_xlabel("PC1")
                            axes[1].set_ylabel("PC2")
                            axes[1].set_title(f"Clusters (k={best_k}) on PCA Space")
                            plt.colorbar(scatter, ax=axes[1], label="Cluster")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Cluster stats
                        with st.expander("Cluster Statistics"):
                            cluster_df = df.loc[pca_idx].copy()
                            cluster_df["Cluster"] = cluster_labels
                            cluster_stats = cluster_df.groupby("Cluster")[numeric_cols].mean().round(2)
                            st.dataframe(cluster_stats, use_container_width=True)
                    else:
                        st.info("Not enough data points for clustering.")
                else:
                    st.info("Need at least 2 numeric columns for clustering.")

        # ============================================================
        # SECTION 5 — Advanced Analysis
        # ============================================================
        st.header(f"🧠 {4 + section_offset}. Advanced Analysis")

        adv_tabs = st.tabs(["🎯 Target Analysis", "📊 Feature Importance", "🔎 Outlier Deep Dive"])

        with adv_tabs[0]:
            st.subheader("Target Variable Analysis")
            target_col = st.selectbox(
                "Select target variable (what you want to predict/analyze):",
                ["(none)"] + numeric_cols + categorical_cols,
                key="target_select"
            )
            if target_col != "(none)":
                if target_col in numeric_cols:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    sns.histplot(df[target_col].dropna(), kde=True, ax=axes[0], color="#e74c3c")
                    axes[0].set_title(f"Target Distribution — {target_col}")
                    # QQ-like check with sorted values
                    sorted_vals = np.sort(df[target_col].dropna().values)
                    axes[1].plot(sorted_vals, marker='.', linestyle='none', alpha=0.5, color="#3498db")
                    axes[1].set_title(f"Sorted Values — {target_col}")
                    axes[1].set_xlabel("Rank")
                    axes[1].set_ylabel(target_col)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Stats
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Mean", f"{df[target_col].mean():.3f}")
                    col_b.metric("Median", f"{df[target_col].median():.3f}")
                    col_c.metric("Std Dev", f"{df[target_col].std():.3f}")

                elif target_col in categorical_cols:
                    vc = df[target_col].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    if len(vc) <= 10:
                        ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%',
                               colors=sns.color_palette("Set2", len(vc)))
                        ax.set_title(f"Target Distribution — {target_col}")
                    else:
                        top_vc = vc.head(10)
                        ax.barh(range(len(top_vc)), top_vc.values,
                               color=sns.color_palette("viridis", len(top_vc)))
                        ax.set_yticks(range(len(top_vc)))
                        ax.set_yticklabels(top_vc.index)
                        ax.set_title(f"Top 10 Categories — {target_col}")
                        ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        with adv_tabs[1]:
            st.subheader("Feature Importance (Correlation-based)")
            if target_col != "(none)" and target_col in numeric_cols:
                importances = df[numeric_cols].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, max(4, len(importances) * 0.35)))
                colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.3 else "#3498db"
                          for v in importances.values]
                ax.barh(range(len(importances)), importances.values, color=colors)
                ax.set_yticks(range(len(importances)))
                ax.set_yticklabels(importances.index)
                ax.set_title(f"Feature Importance for '{target_col}' (|correlation|)")
                ax.set_xlabel("Absolute Correlation")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Table
                imp_df = pd.DataFrame({
                    "Feature": importances.index,
                    "|Correlation|": importances.values.round(3),
                    "Strength": ["🔴 Strong" if v > 0.5 else "🟡 Moderate" if v > 0.3 else "🔵 Weak"
                                for v in importances.values]
                })
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Select a numeric target variable above to see feature importance.")

        with adv_tabs[2]:
            st.subheader("Outlier Deep Dive")
            if numeric_cols:
                outlier_col = st.selectbox("Select column:", numeric_cols, key="outlier_col")
                q1 = df[outlier_col].quantile(0.25)
                q3 = df[outlier_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]

                col_x, col_y, col_z = st.columns(3)
                col_x.metric("Total Outliers", len(outliers))
                col_y.metric("Lower Bound", f"{lower_bound:.2f}")
                col_z.metric("Upper Bound", f"{upper_bound:.2f}")

                if len(outliers) > 0:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    # Highlight outliers in scatter
                    normal = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                    axes[0].scatter(range(len(normal)), normal[outlier_col], alpha=0.3,
                                   s=10, color="#3498db", label="Normal")
                    axes[0].scatter(range(len(normal), len(normal) + len(outliers)),
                                   outliers[outlier_col], alpha=0.7, s=20,
                                   color="#e74c3c", label="Outliers")
                    axes[0].axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.7)
                    axes[0].axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.7)
                    axes[0].legend()
                    axes[0].set_title(f"Outliers in {outlier_col}")

                    # Box plot with outliers
                    sns.boxplot(x=df[outlier_col], ax=axes[1], color="#2ecc71")
                    axes[1].set_title(f"Box Plot — {outlier_col}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    with st.expander(f"View outlier rows ({len(outliers)} rows)"):
                        st.dataframe(outliers.head(50), use_container_width=True)
            else:
                st.info("No numerical columns for outlier analysis.")



        # ============================================================
        # SECTION — AI Auto-Summary (NEW)
        # ============================================================
        if run_auto_summary:
            st.header(f"🤖 {5 + section_offset}. AI Executive Summary")
            summary_key = f"auto_summary_{uploaded_file.name}"
            if summary_key not in st.session_state:
                summary_context = build_data_context(
                    df=df, uploaded_name=uploaded_file.name,
                    numeric_cols=numeric_cols, categorical_cols=categorical_cols,
                    engineered_features=engineered_features if run_feature_eng else None,
                    imputation_report=imputation_report if run_feature_eng else None,
                    skew_df=skew_df if run_feature_eng else None,
                    date_cols=date_cols,
                )

                if not selected_model.startswith("("):
                    with st.spinner(f"Generating executive summary with {selected_model}..."):
                        summary_prompt = (
                            f"You are a senior data analyst. Analyze this dataset and provide a concise executive summary.\n\n"
                            f"{summary_context}\n\n"
                            f"Provide:\n"
                            f"1. **Data Quality Grade** (A-F) with justification\n"
                            f"2. **Top 3 Key Patterns** with specific numbers\n"
                            f"3. **Top 3 Actionable Recommendations** with concrete steps\n"
                            f"4. **Predicted Best Use Cases** for this dataset\n"
                            f"5. **Red Flags** (any concerns about data quality or validity)\n"
                            f"Be specific and cite exact numbers."
                        )
                        timeout = TIMEOUT_CLOUD if selected_model in CLOUD_MODELS else TIMEOUT_LOCAL
                        summary_response = query_ollama(selected_model, summary_prompt, system=SYSTEM_ROLE, timeout=timeout)
                        st.session_state[summary_key] = summary_response
                else:
                    st.session_state[summary_key] = "❌ No model available."

            st.markdown(st.session_state[summary_key])
            if st.button("🔄 Regenerate Summary", key="regen_summary"):
                if summary_key in st.session_state:
                    del st.session_state[summary_key]
                    st.rerun()

        # ============================================================
        # SECTION — AI Q&A: Ask Questions, Get Solutions
        # ============================================================
        qa_section_num = 5 + section_offset + (1 if run_auto_summary else 0)
        st.header(f"💡 {qa_section_num}. AI-Powered Q&A — Ask Anything About Your Data")

        st.markdown(
            "Ask any question about your data and get **detailed answers with solutions, "
            "insights, and actionable advice** from your AI model."
        )
        if run_consensus:
            st.caption("🔀 **Multi-Model Consensus** is ON — your question will be sent to multiple models.")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            # Show user message
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.chat_history.append({"role": "user", "content": user_question})

            # Build context with smart truncation
            data_context = build_data_context(
                df=df,
                uploaded_name=uploaded_file.name,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                engineered_features=engineered_features if run_feature_eng else None,
                imputation_report=imputation_report if run_feature_eng else None,
                skew_df=skew_df if run_feature_eng else None,
                date_cols=date_cols,
                user_question=user_question,
            )

            # Smart context truncation — local models get generous context with num_ctx=8192
            max_context = 4000 if selected_model in CLOUD_MODELS else 3500
            if len(data_context) > max_context:
                data_context = data_context[:max_context] + "\n... [context truncated for model context window]"

            # Previous conversation context
            prev_context = ""
            if len(st.session_state.chat_history) > 2:
                recent = st.session_state.chat_history[-6:]
                prev_context = "\n## PREVIOUS CONVERSATION\n"
                for msg in recent[:-1]:
                    role = "USER" if msg["role"] == "user" else "ANALYST"
                    prev_context += f"{role}: {msg['content'][:500]}\n"

            # Enhanced chain-of-thought prompt
            prompt = textwrap.dedent(f"""\
You are a senior data analyst. Think through this step-by-step before answering.

## STEP 1: UNDERSTAND THE QUESTION
Re-state the user's question in your own words to ensure understanding.

## STEP 2: IDENTIFY RELEVANT DATA
Which specific columns, rows, or statistics are most relevant?

## STEP 3: ANALYZE
Apply appropriate analytical techniques (comparison, trend analysis, statistics).

## USER'S QUESTION
{user_question}

## FULL DATA ANALYSIS RESULTS
{data_context}
{prev_context}

## YOUR RESPONSE FORMAT:

### 📌 DIRECT ANSWER
Answer with SPECIFIC numbers, names, and values. If asking about specific items, list them.

### 🔍 KEY INSIGHTS (3-5)
Important patterns with actual numbers. Explain WHY, not just WHAT.

### ✅ SOLUTION & RECOMMENDATIONS
Step-by-step, actionable. Include thresholds, filters, code snippets where useful.

### ⚠️ DATA QUALITY WARNINGS
Flag issues that might affect the answer. Quantify the impact.

### 💡 FOLLOW-UP ANALYSIS
What should be investigated next? Be specific.

### 🎯 CONFIDENCE LEVEL
Rate your confidence (High/Medium/Low) and explain what would change your answer.

CRITICAL: Use SPECIFIC numbers. NO vague statements. Be concrete and actionable.
""")

            if selected_model.startswith("("):
                with st.chat_message("assistant"):
                    st.error("No Ollama model available. Install one with `ollama pull <model>`.")
            elif run_consensus:
                # Multi-model consensus mode
                valid_models = [m for m in available_models if not m.startswith("(")][:3]
                with st.chat_message("assistant"):
                    if len(valid_models) < 2:
                        st.warning("Need at least 2 models for consensus. Falling back to single model.")
                        response = st.write_stream(
                            stream_ollama(selected_model, prompt, system=SYSTEM_ROLE)
                        )
                    else:
                        responses = {}
                        prev_mdl = None
                        for mdl in valid_models:
                            # Unload previous model to free memory before loading next
                            if prev_mdl and prev_mdl not in CLOUD_MODELS:
                                unload_ollama_model(prev_mdl)
                            mdl_label = f"☁️ {mdl}" if mdl in CLOUD_MODELS else f"💻 {mdl}"
                            with st.spinner(f"Querying {mdl_label}..."):
                                timeout = TIMEOUT_CLOUD if mdl in CLOUD_MODELS else TIMEOUT_LOCAL
                                responses[mdl] = query_ollama(mdl, prompt, system=SYSTEM_ROLE, timeout=timeout)
                            prev_mdl = mdl
                        # Unload last model
                        if prev_mdl and prev_mdl not in CLOUD_MODELS:
                            unload_ollama_model(prev_mdl)

                        # Show individual responses in expanders
                        for mdl, resp in responses.items():
                            mdl_label = f"☁️ {mdl}" if mdl in CLOUD_MODELS else f"💻 {mdl}"
                            with st.expander(f"Response from {mdl_label}", expanded=False):
                                st.markdown(resp)

                        # Synthesize consensus using best model
                        synthesis_prompt = (
                            f"You received answers from {len(responses)} AI models to the question: '{user_question}'\n\n"
                        )
                        for mdl, resp in responses.items():
                            synthesis_prompt += f"### Model: {mdl}\n{resp[:1500]}\n\n"
                        synthesis_prompt += (
                            "\nSynthesize a CONSENSUS answer:\n"
                            "1. What do all models AGREE on? (highlight these as high-confidence findings)\n"
                            "2. Where do they DISAGREE? (explain the different perspectives)\n"
                            "3. What is the BEST answer combining all insights?\n"
                            "Use specific numbers and data references."
                        )
                        st.markdown("---")
                        st.markdown("### 🔀 Multi-Model Consensus")
                        response = st.write_stream(
                            stream_ollama(selected_model, synthesis_prompt, system=SYSTEM_ROLE)
                        )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
            else:
                # Single model mode
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        stream_ollama(selected_model, prompt, system=SYSTEM_ROLE)
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

        # Show a prompt if no questions yet
        if not st.session_state.chat_history:
            st.info(
                "💬 **Try asking:**\n"
                "- *What are the main patterns in this data?*\n"
                "- *Which features are most important for predicting X?*\n"
                "- *Are there any anomalies or outliers I should worry about?*\n"
                "- *What predictions can I make from this data?*\n"
                "- *How should I clean this dataset for machine learning?*\n"
                "- *Compare the top 5 categories by revenue and growth rate*"
            )

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.exception(e)

else:
    # Landing page when no file is uploaded
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 📁 Upload")
        st.markdown("Drop any CSV or Excel file into the sidebar to get started.")
    with col_b:
        st.markdown("### 🔬 Analyze")
        st.markdown("Full EDA, feature engineering, correlations, outlier detection & more.")
    with col_c:
        st.markdown("### 💬 Ask")
        st.markdown("Chat with your data — get detailed solutions powered by local AI.")
    st.info("👈 Upload a dataset from the sidebar to begin.")
