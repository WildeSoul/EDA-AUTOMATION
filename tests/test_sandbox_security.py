"""
Security and Isolation Tests for Safe Code Execution Sandbox.
Validates all Phase 1 acceptance criteria: process isolation, timeouts, memory caps,
network/filesystem defenses, object traversal prevention, and stdout truncation.
"""
import os
import sys
import threading
import time
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.code_sandbox import execute_code_safely, validate_code_ast


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 75000, 90000, 110000],
        "department": ["IT", "HR", "IT", "Sales", "HR"],
    })


# ═══════════════════════════════════════════════
# 1. Normal Pandas & Matplotlib Execution
# ═══════════════════════════════════════════════

def test_normal_pandas_execution(sample_df):
    """Test valid pandas computations execute and capture result DataFrame."""
    code = """
result = df.groupby('department')['salary'].mean().reset_index()
print("Groupby completed successfully")
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=10)
    assert res["success"] is True, f"Execution failed: {res.get('error')}"
    assert "Groupby completed successfully" in res["stdout"]
    assert res["result_df"] is not None
    assert len(res["result_df"]) == 3
    assert "salary" in res["result_df"].columns


def test_matplotlib_visualization(sample_df):
    """Test matplotlib figure creation is captured and reconstructed."""
    code = """
plt.figure(figsize=(5, 3))
plt.bar(df['department'], df['salary'])
plt.title('Salary by Dept')
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=10)
    assert res["success"] is True, f"Execution failed: {res.get('error')}"
    assert len(res["figures"]) == 1
    assert hasattr(res["figures"][0], "savefig")


# ═══════════════════════════════════════════════
# 2. Infinite Loop & Wall-Clock Timeout
# ═══════════════════════════════════════════════

def test_infinite_loop_timeout(sample_df):
    """Test that an infinite loop is terminated by the wall-clock timeout."""
    code = """
x = 0
while True:
    x += 1
"""
    # Use 2 seconds timeout for fast test execution
    res = execute_code_safely(code, sample_df, timeout_seconds=2)
    assert res["success"] is False
    assert "ExecutionTimeout" in res["error"]
    assert "terminated" in res["error"]


# ═══════════════════════════════════════════════
# 3. Memory Exhaustion Termination
# ═══════════════════════════════════════════════

def test_memory_exhaustion(sample_df):
    """Test that allocating excessive memory terminates the worker without crashing."""
    code = """
# Attempt to allocate 400 MB of data when cap is 100 MB
big_list = [b'x' * (1024 * 1024) for _ in range(400)]
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=10, max_memory_mb=100)
    assert res["success"] is False
    assert "MemoryLimitExceeded" in res["error"] or "WorkerCrash" in res["error"] or "terminated" in res["error"]


# ═══════════════════════════════════════════════
# 4. Filesystem Access Restrictions
# ═══════════════════════════════════════════════

def test_filesystem_write_denied(sample_df):
    """Test that file write/creation attempts are denied."""
    code = """
with open('test_malicious_write.txt', 'w') as f:
    f.write('hacked')
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "forbidden" in res["error"].lower() or "permissionerror" in res["error"].lower()
    assert not os.path.exists('test_malicious_write.txt')


# ═══════════════════════════════════════════════
# 5. Network Access Restrictions
# ═══════════════════════════════════════════════

def test_network_access_denied(sample_df):
    """Test that outbound socket connections are blocked."""
    code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('1.1.1.1', 80))
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "network access is blocked" in res["error"].lower() or "security violation" in res["error"].lower()


# ═══════════════════════════════════════════════
# 6. Object Traversal & Introspection Escapes
# ═══════════════════════════════════════════════

def test_object_traversal_blocked(sample_df):
    """Test that Python object introspection (e.g. __subclasses__) is caught and blocked."""
    code = """
classes = ().__class__.__base__.__subclasses__()
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "Security Violation" in res["error"]
    assert "__subclasses__" in res["error"] or "__class__" in res["error"]


def test_eval_exec_calls_blocked(sample_df):
    """Test that nested eval() or exec() calls are rejected by AST validation."""
    code = """
eval("1 + 1")
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "eval()" in res["error"]


# ═══════════════════════════════════════════════
# 7. Stdout Truncation & Output Capping
# ═══════════════════════════════════════════════

def test_excessive_stdout_capped(sample_df):
    """Test that excessive stdout output is capped at the configured limit."""
    code = """
for i in range(1500):
    print("This is a very long line of spam output that exceeds buffer caps")
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=15)
    assert res["success"] is True, f"Execution failed: {res.get('error')}"
    # Default cap is 50 KB = 51,200 bytes
    assert len(res["stdout"]) <= 60000
    assert "Output capped at" in res["stdout"]


# ═══════════════════════════════════════════════
# 8. Worker Crash Resilience
# ═══════════════════════════════════════════════

def test_worker_crash_resilience(sample_df):
    """Test that hard worker termination (e.g. os._exit) leaves parent operational."""
    code = """
import os
os._exit(42)
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    # AST blocks os import, or if bypassed, parent catches WorkerCrash
    assert "Security Violation" in res["error"] or "WorkerCrash" in res["error"] or "Import of 'os' is forbidden" in res["error"]


# ═══════════════════════════════════════════════
# 9. Filesystem Read Isolation & Subprocess Containment
# ═══════════════════════════════════════════════

def test_sensitive_file_read_denied(sample_df):
    """Test that reading host files, system configs, or .env is denied."""
    code = """
with open('.env', 'r') as f:
    secret = f.read()
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "forbidden" in res["error"].lower() or "security violation" in res["error"].lower()


def test_subprocess_spawn_denied(sample_df):
    """Test that spawning child subprocesses is strictly prohibited."""
    code = """
import subprocess
subprocess.Popen(['notepad.exe'])
"""
    res = execute_code_safely(code, sample_df, timeout_seconds=5)
    assert res["success"] is False
    assert "forbidden" in res["error"].lower() or "security violation" in res["error"].lower()


def test_sandbox_active_cancellation(sample_df):
    """Test that an active sandbox computation is terminated immediately upon cancel signal."""
    code = """
x = 0
for i in range(100000000):
    x += 1
"""
    cancel_ev = threading.Event()
    # Trigger cancellation in a separate thread after 200ms
    def trigger_cancel():
        time.sleep(0.2)
        cancel_ev.set()

    t = threading.Thread(target=trigger_cancel)
    t.start()

    res = execute_code_safely(code, sample_df, timeout_seconds=10, cancel_event=cancel_ev)
    t.join()

    assert res["success"] is False
    assert "ExecutionCancelled" in res["error"]

