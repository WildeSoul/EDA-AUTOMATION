import os
import json
import pandas as pd
from pathlib import Path

def test_leaderboard_file_structure():
    """Verify that leaderboard.json exists and contains correct keys and datasets."""
    leaderboard_path = Path(__file__).parent.parent / "evaluation" / "leaderboard.json"
    assert leaderboard_path.exists(), "leaderboard.json is missing!"
    
    with open(leaderboard_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    required_keys = [
        "dataset_intelligence_accuracy",
        "target_detection_accuracy",
        "semantic_typing_accuracy",
        "nl_sql_accuracy",
        "sql_execution_success"
    ]
    for key in required_keys:
        assert key in data, f"Key {key} not found in leaderboard.json"
        assert isinstance(data[key], (int, float)), f"Key {key} must be a number"
        
    assert "benchmark_datasets" in data, "benchmark_datasets missing in leaderboard"
    assert "Titanic" in data["benchmark_datasets"]
    assert "Credit Risk" in data["benchmark_datasets"]
    assert "Employee Attrition" in data["benchmark_datasets"]
    assert "Patient Treatment" in data["benchmark_datasets"]

def test_demo_datasets_exist():
    """Verify that all 6 demo datasets exist and can be parsed by pandas."""
    demo_dir = Path(__file__).parent.parent / "data" / "demo"
    assert demo_dir.exists(), "demo directory is missing!"
    
    demo_files = [
        "titanic.csv",
        "customer_churn.csv",
        "sales_transactions.csv",
        "credit_risk.csv",
        "employee_attrition.csv",
        "patient_treatment.csv"
    ]
    for file in demo_files:
        path = demo_dir / file
        assert path.exists(), f"Demo file {file} is missing!"
        
        # Test parsing
        df = pd.read_csv(path)
        assert len(df) > 0, f"Demo file {file} is empty!"
        assert len(df.columns) > 0, f"Demo file {file} has no columns!"
