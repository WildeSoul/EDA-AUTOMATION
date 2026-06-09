"""
Natural Language → Visualization — detect chart intent and generate plots.
"""
import re
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from core.llm import query_ollama
from tools.code_sandbox import execute_code_safely
from config.settings import (
    SYSTEM_ROLE, TIMEOUT_CLOUD, TIMEOUT_LOCAL, CLOUD_MODELS, NEMOTRON_MODELS,
)

# Keywords that suggest visualization intent
VIZ_KEYWORDS = {
    "plot", "chart", "graph", "visualize", "visualization", "show me",
    "draw", "histogram", "scatter", "bar chart", "pie chart", "heatmap",
    "boxplot", "box plot", "violin", "line chart", "trend", "distribution",
}


def detect_viz_intent(question: str) -> bool:
    """Check if the user's question is asking for a visualization."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in VIZ_KEYWORDS)


def generate_visualization(
    question: str,
    df: pd.DataFrame,
    model: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Generate a visualization from a natural language request.

    Returns dict with:
        success: bool
        figures: list[plt.Figure]
        code: str (the generated code)
        error: str or None
    """
    result = {"success": False, "figures": [], "code": "", "error": None}

    dtypes_str = df.dtypes.to_string()
    sample_str = df.head(3).to_string(index=False)

    prompt = textwrap.dedent(f"""\
    Generate ONLY Python matplotlib/seaborn code to create a visualization.

    ## DATAFRAME INFO
    Columns and types:
    {dtypes_str}

    Numeric columns: {numeric_cols[:15]}
    Categorical columns: {categorical_cols[:10]}

    Sample data:
    {sample_str}

    ## VISUALIZATION REQUEST
    {question}

    ## RULES
    1. Output ONLY executable Python code — NO markdown, NO explanations, NO ```
    2. `df`, `pd`, `np`, `plt`, `sns`, `matplotlib` are already imported
    3. Use plt.figure(figsize=(12, 6)) for a good chart size
    4. Add title, axis labels, and legend where appropriate
    5. Use a modern color palette (e.g., sns.color_palette("husl"))
    6. Call plt.tight_layout() at the end
    7. Handle missing values with .dropna()
    8. For categorical data with many values, limit to top 15

    ## YOUR CODE:
    """)

    viz_system = (
        "You are a data visualization expert. Generate ONLY executable Python "
        "matplotlib/seaborn code. No explanations, no markdown fences."
    )

    # Prefer specialized code-gen model, fallback to user-selected model
    code_model = NEMOTRON_MODELS.get("code_gen", model)
    timeout = TIMEOUT_CLOUD if code_model in CLOUD_MODELS else TIMEOUT_LOCAL
    code = query_ollama(code_model, prompt, system=viz_system, timeout=timeout)

    # Clean up: remove markdown code fences if LLM included them
    code = re.sub(r"^```python\s*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE)
    code = code.strip()

    if not code or code.startswith("❌"):
        result["error"] = code or "No code generated."
        return result

    result["code"] = code

    # Execute the generated code
    exec_result = execute_code_safely(code, df)

    if exec_result["success"] and exec_result["figures"]:
        result["success"] = True
        result["figures"] = exec_result["figures"]
    elif exec_result["error"]:
        result["error"] = exec_result["error"]
    else:
        result["error"] = "Code executed but no figure was created."

    return result
