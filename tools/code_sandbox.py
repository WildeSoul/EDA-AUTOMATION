"""
Safe Code Execution Sandbox — executes LLM-generated Python code
against the user's DataFrame with restricted builtins.
"""
import io
import sys
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from config.settings import SANDBOX_ALLOWED_BUILTINS


def _build_safe_globals(df: pd.DataFrame) -> dict:
    """Build a restricted globals dict for exec()."""
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in SANDBOX_ALLOWED_BUILTINS
                     if (isinstance(__builtins__, dict) and k in __builtins__) or
                        (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))}

    return {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "matplotlib": matplotlib,
    }


def execute_code_safely(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = 30,
) -> dict:
    """
    Execute Python code in a sandboxed environment.

    Returns dict with:
        success: bool
        stdout: str (captured print output)
        result_df: pd.DataFrame or None (if code assigns to 'result')
        figures: list[plt.Figure] (captured matplotlib figures)
        error: str or None
    """
    result = {
        "success": False,
        "stdout": "",
        "result_df": None,
        "figures": [],
        "error": None,
    }

    # Close any existing figures
    plt.close("all")

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        safe_globals = _build_safe_globals(df)
        safe_locals = {}

        # Execute the code
        exec(code, safe_globals, safe_locals)

        # Capture stdout
        result["stdout"] = captured_output.getvalue()

        # Check for result DataFrame
        if "result" in safe_locals and isinstance(safe_locals["result"], pd.DataFrame):
            result["result_df"] = safe_locals["result"]

        # Capture any matplotlib figures that were created
        figures = [plt.figure(n) for n in plt.get_fignums()]
        result["figures"] = figures

        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return result


def generate_code_prompt(user_request: str, df: pd.DataFrame, columns_info: str) -> str:
    """Build a prompt that instructs the LLM to generate executable pandas code."""
    sample = df.head(3).to_string(index=False)
    dtypes = df.dtypes.to_string()

    return f"""You are a Python data analyst. Generate ONLY executable Python code.

## AVAILABLE VARIABLES
- `df`: pandas DataFrame (already loaded)
- `pd`, `np`, `plt`, `sns`, `matplotlib` are available

## DATAFRAME INFO
Columns and types:
{dtypes}

Sample rows:
{sample}

Column details:
{columns_info}

## USER REQUEST
{user_request}

## RULES
1. Output ONLY Python code — no markdown, no explanations, no ```python blocks
2. Use `df` directly — it's already loaded
3. For visualizations: use plt.figure(), plt.title(), plt.tight_layout()
4. For tabular results: assign to a variable called `result` (as a DataFrame)
5. Use print() for text output
6. Do NOT use os, sys, subprocess, open(), or file operations
7. Do NOT modify the original df — work on copies if needed
8. Handle missing values gracefully with .dropna() or .fillna()

## YOUR CODE (Python only, no markdown):
"""
