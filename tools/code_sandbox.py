"""
Safe Code Execution Sandbox — executes LLM-generated Python code
in an isolated subprocess with strict OS-level resource limits,
network/filesystem restrictions, memory governance, and wall-clock timeouts.
"""
from __future__ import annotations

import ast
import base64
import ctypes
import io
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from config.settings import (
    SANDBOX_ALLOWED_BUILTINS,
    SANDBOX_ALLOWED_MODULES,
    SANDBOX_MAX_FIGURES,
    SANDBOX_MAX_MEMORY_MB,
    SANDBOX_MAX_RESULT_ROWS,
    SANDBOX_MAX_STDOUT_BYTES,
    SANDBOX_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# ── AST Security Validator ────────────────────────────────────

_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
_FORBIDDEN_ATTRS = {
    "__subclasses__", "__class__", "__base__", "__bases__",
    "__globals__", "__code__", "__closure__", "__builtins__",
}


def validate_code_ast(code: str) -> str | None:
    """Scan code AST for forbidden patterns (eval, exec, object-traversal).

    Returns None if valid, or an error message string if forbidden construct found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError in code: {e}"

    for node in ast.walk(tree):
        # Disallow eval, exec, compile calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                return f"Security Violation: '{node.func.id}()' is strictly forbidden in the sandbox."
        # Disallow traversal into __subclasses__, __class__, etc.
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS:
                return f"Security Violation: Access to '{node.attr}' is forbidden in the sandbox."
        # Disallow dynamic imports of unwhitelisted modules
        elif isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in SANDBOX_ALLOWED_MODULES:
                    return f"Security Violation: Import of '{alias.name}' is forbidden in the sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base not in SANDBOX_ALLOWED_MODULES:
                    return f"Security Violation: Import from '{node.module}' is forbidden in the sandbox."

    return None


class LimitedStringIO(io.StringIO):
    """StringIO buffer that caps written data at max_bytes to prevent memory exhaustion."""

    def __init__(self, max_bytes: int):
        super().__init__()
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.truncated = False

    def write(self, s: str) -> int:
        if self.truncated:
            return 0
        s_bytes = len(s.encode("utf-8", errors="replace"))
        if self.current_bytes + s_bytes > self.max_bytes:
            allowed = max(0, self.max_bytes - self.current_bytes)
            super().write(s[:allowed])
            super().write(f"\n... [Output capped at {self.max_bytes // 1024} KB limit]")
            self.truncated = True
            return allowed
        self.current_bytes += s_bytes
        return super().write(s)


def _sandbox_worker_entry(
    code: str,
    df: pd.DataFrame,
    result_queue: Any,
    max_stdout_bytes: int,
    max_figures: int,
    max_result_rows: int,
) -> None:
    """Entry point executed inside the isolated child process."""
    result: dict[str, Any] = {
        "success": False,
        "stdout": "",
        "result_df": None,
        "figures_png_bytes": [],
        "error": None,
    }

    # 1. Pre-load standard libraries & build safe namespace BEFORE hooking imports
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.close("all")
    import builtins
    import seaborn as sns
    import PIL.Image

    # Pre-initialize PIL and matplotlib engines
    try:
        PIL.Image.init()
    except Exception:
        pass

    def _sandbox_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "+", "x")):
            raise PermissionError(f"Filesystem write access is forbidden in the sandbox: open('{file}', '{mode}')")
        raise PermissionError(f"Direct filesystem access is forbidden in the sandbox: open('{file}', '{mode}')")

    safe_builtins = {
        k: builtins.__dict__[k] if isinstance(builtins, type(sys)) else getattr(builtins, k)
        for k in SANDBOX_ALLOWED_BUILTINS
        if hasattr(builtins, k)
    }
    safe_builtins["open"] = _sandbox_open

    safe_globals = {
        "__name__": "__sandbox__",
        "__builtins__": safe_builtins,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "matplotlib": matplotlib,
    }
    safe_locals: dict[str, Any] = {}

    # 2. Install Network Isolation in Child Process
    try:
        import socket
        _orig_socket = socket.socket

        class _SandboxedSocket(_orig_socket):
            def __init__(self, *args, **kwargs):
                raise PermissionError("Network access is blocked in the sandbox environment.")

        socket.socket = _SandboxedSocket
        socket.create_connection = lambda *a, **kw: (_ for _ in ()).throw(
            PermissionError("Network access is blocked in the sandbox environment.")
        )
    except Exception as e:
        logger.debug("Failed installing socket block: %s", e)

    # 3. Install Import Restriction in Child Process
    _ALLOWED_ROOTS = {
        "builtins", "math", "cmath", "datetime", "time", "json", "re",
        "collections", "itertools", "functools", "operator", "copy",
        "types", "typing", "typing_extensions", "enum", "numbers",
        "decimal", "fractions", "io", "_io", "sys", "contextlib", "abc",
        "weakref", "string", "unicodedata", "warnings", "traceback",
        "encodings", "linecache", "IPython", "ipython", "__future__",
        "pandas", "numpy", "matplotlib", "mpl_toolkits", "seaborn",
        "scipy", "sklearn", "statsmodels",
        "pytz", "dateutil", "six", "PIL", "kiwisolver", "fontTools",
        "packaging", "contourpy", "cycler",
    }
    _FORBIDDEN_ROOTS = {
        "os", "subprocess", "shutil", "socket", "urllib", "requests",
        "http", "ftplib", "smtplib", "imaplib", "poplib", "telnetlib",
        "xmlrpc", "asyncio", "ctypes", "winreg", "msvcrt",
        "multiprocessing", "threading", "pty", "posix", "nt",
        "importlib", "pip", "webbrowser", "pathlib",
    }
    try:
        _orig_import = builtins.__import__

        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Check if this import originates from within an already allowed library (e.g. PIL, matplotlib)
            if globals:
                caller_pkg = globals.get("__package__") or globals.get("__name__", "")
                caller_root = caller_pkg.split(".")[0] if caller_pkg else ""
                if caller_root in _ALLOWED_ROOTS and caller_root != "__sandbox__":
                    return _orig_import(name, globals, locals, fromlist, level)

            base = name.split(".")[0]
            if base in _FORBIDDEN_ROOTS or base not in _ALLOWED_ROOTS:
                raise PermissionError(f"Importing module '{name}' is forbidden in the sandbox.")
            return _orig_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _restricted_import
    except Exception as e:
        logger.debug("Failed installing import hook: %s", e)

    # 4. Install Filesystem Restriction in Child Process
    try:
        builtins.open = _sandbox_open
    except Exception as e:
        logger.debug("Failed installing open hook: %s", e)

    # 5. Capture stdout & Execute User Code
    captured_output = LimitedStringIO(max_stdout_bytes)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_output
    sys.stderr = captured_output

    try:
        # Execute code in isolated namespace
        exec(code, safe_globals, safe_locals)

        result["stdout"] = captured_output.getvalue()

        # Capture result DataFrame if assigned
        if "result" in safe_locals and isinstance(safe_locals["result"], pd.DataFrame):
            res_df = safe_locals["result"]
            if len(res_df) > max_result_rows:
                res_df = res_df.head(max_result_rows)
            result["result_df"] = res_df

        # Capture matplotlib figures as raw PNG bytes
        figures_bytes = []
        fig_nums = plt.get_fignums()[:max_figures]
        for n in fig_nums:
            fig = plt.figure(n)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            figures_bytes.append(buf.getvalue())

        result["figures_png_bytes"] = figures_bytes
        result["success"] = True

    except Exception as e:
        result["stdout"] = captured_output.getvalue()
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            result_queue.put(result)
        except Exception as q_err:
            logger.error("Failed putting worker result in queue: %s", q_err)


def _kill_process_tree(pid: int) -> None:
    """Forcibly terminate a process and all its child subprocesses."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


# ── Windows Job Object structures for OS-level resource governance ─
class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryLimit", ctypes.c_size_t),
        ("PeakJobMemoryLimit", ctypes.c_size_t),
    ]


def _assign_process_to_job(pid: int, max_memory_mb: int) -> int | None:
    """Create a Windows Job Object with KILL_ON_JOB_CLOSE and assign child process."""
    try:
        kernel32 = ctypes.windll.kernel32
        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            return None

        # 0x2000 = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        # Guarantees Windows kernel terminates all child processes if parent exits or crashes
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000

        kernel32.SetInformationJobObject(
            hJob, 9, ctypes.byref(info), ctypes.sizeof(info)
        )

        PROCESS_ALL_ACCESS = 0x1F0FFF
        hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if hProcess:
            kernel32.AssignProcessToJobObject(hJob, hProcess)
            kernel32.CloseHandle(hProcess)
            return hJob
    except Exception as e:
        logger.debug("Failed configuring Windows Job Object: %s", e)
    return None


def _close_job_handle(hJob: int | None) -> None:
    """Safely close a Windows Job Object handle."""
    if hJob:
        try:
            ctypes.windll.kernel32.CloseHandle(hJob)
        except Exception:
            pass


def _bytes_to_figure(png_bytes: bytes) -> plt.Figure:
    """Reconstruct a matplotlib Figure from serialized PNG bytes in the parent process."""
    fig, ax = plt.subplots(figsize=(6, 4))
    img = mpimg.imread(io.BytesIO(png_bytes))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def execute_code_safely(
    code: str,
    df: pd.DataFrame,
    timeout_seconds: int = SANDBOX_TIMEOUT_SECONDS,
    max_memory_mb: int = SANDBOX_MAX_MEMORY_MB,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    """Execute Python code in an isolated child process with hard resource limits.

    Guarantees:
      - Worker termination on timeout (never hangs main process)
      - Worker termination on OOM / memory limit breach (never crashes main process)
      - Active termination upon cancel_event signaling
      - Network / filesystem write restrictions
      - Output and figure size caps
      - Returns standard contract: {success, stdout, result_df, figures, error}
    """
    # Step 1: Pre-execution AST check
    ast_error = validate_code_ast(code)
    if ast_error:
        return {
            "success": False,
            "stdout": "",
            "result_df": None,
            "figures": [],
            "error": ast_error,
        }

    # Step 2: Spawn isolated child process
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    worker = ctx.Process(
        target=_sandbox_worker_entry,
        args=(
            code,
            df,
            result_queue,
            SANDBOX_MAX_STDOUT_BYTES,
            SANDBOX_MAX_FIGURES,
            SANDBOX_MAX_RESULT_ROWS,
        ),
    )

    start_time = time.time()
    worker.start()

    # Step 3: Optional Windows Job Object attachment
    job_handle = None
    if sys.platform == "win32":
        try:
            job_handle = _assign_process_to_job(worker.pid, max_memory_mb)
        except Exception as job_err:
            logger.debug("Windows Job Object assignment skipped/failed: %s", job_err)

    # Step 4: Active Monitoring Loop (Timeout, Memory Cap, Cancel Event, & Queue Retrieval)
    max_memory_bytes = max_memory_mb * 1024 * 1024
    poll_interval = 0.05
    timed_out = False
    oom_killed = False
    raw_result = None

    try:
        proc = psutil.Process(worker.pid)
    except (psutil.NoSuchProcess, Exception):
        proc = None

    while True:
        # Check active external cancellation
        if cancel_event is not None and cancel_event.is_set():
            _kill_process_tree(worker.pid)
            worker.join(timeout=1.0)
            _close_job_handle(job_handle)
            return {
                "success": False,
                "stdout": "",
                "result_df": None,
                "figures": [],
                "error": "ExecutionCancelled: Worker process terminated by cancellation request.",
            }

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            timed_out = True
            break

        # Check memory
        if proc is not None:
            try:
                mem_info = proc.memory_info()
                if mem_info.rss > max_memory_bytes:
                    oom_killed = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Try to retrieve result from queue without deadlocking
        try:
            raw_result = result_queue.get(timeout=poll_interval)
            break
        except queue.Empty:
            pass

        # If worker has died and queue check above returned nothing, check once more
        if not worker.is_alive():
            try:
                raw_result = result_queue.get_nowait()
            except queue.Empty:
                pass
            break

    # Step 5: Termination if limits breached
    if timed_out or oom_killed:
        _kill_process_tree(worker.pid)
        worker.join(timeout=1.0)
        _close_job_handle(job_handle)

        if timed_out:
            return {
                "success": False,
                "stdout": "",
                "result_df": None,
                "figures": [],
                "error": f"ExecutionTimeout: Worker process exceeded {timeout_seconds}s limit and was terminated.",
            }
        else:
            return {
                "success": False,
                "stdout": "",
                "result_df": None,
                "figures": [],
                "error": f"MemoryLimitExceeded: Worker exceeded {max_memory_mb} MB limit and was terminated.",
            }

    # Step 6: Worker exited naturally or crashed
    worker.join(timeout=2.0)
    _close_job_handle(job_handle)

    # Step 7: Validate retrieved result
    if raw_result is None:
        if worker.exitcode != 0:
            return {
                "success": False,
                "stdout": "",
                "result_df": None,
                "figures": [],
                "error": f"WorkerCrash: Worker process terminated unexpectedly with exit code {worker.exitcode}.",
            }
        return {
            "success": False,
            "stdout": "",
            "result_df": None,
            "figures": [],
            "error": "WorkerExit: No result was returned from the isolated worker process.",
        }

    # Step 8: Reconstruct figures for consumer compatibility
    figures = []
    for png_bytes in raw_result.get("figures_png_bytes", []):
        try:
            fig = _bytes_to_figure(png_bytes)
            figures.append(fig)
        except Exception as fig_err:
            logger.warning("Failed reconstructing figure from worker: %s", fig_err)

    return {
        "success": raw_result.get("success", False),
        "stdout": raw_result.get("stdout", ""),
        "result_df": raw_result.get("result_df"),
        "figures": figures,
        "error": raw_result.get("error"),
    }


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
