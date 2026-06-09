import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from data.intelligence import (
    compute_fingerprint,
    _classify_columns_heuristic,
    build_dataset_intelligence
)

def test_compute_fingerprint():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df3 = pd.DataFrame({"a": [1, 2, 4], "b": ["x", "y", "z"]}) # different cell value
    
    fp1 = compute_fingerprint(df1)
    fp2 = compute_fingerprint(df2)
    fp3 = compute_fingerprint(df3)
    
    assert fp1 == fp2
    assert fp1 != fp3

def test_heuristic_classification():
    df = pd.DataFrame({
        "id_col": list(range(101, 125)),
        "churn": [0, 1] * 12,
        "numeric_val": [float(x % 12) for x in range(24)],
        "text_val": [f"val_{x % 10}" for x in range(24)],
        "date_val": [f"2026-01-01" for _ in range(24)]
    })
    
    # Force type conversion or mock date cols
    df["date_val"] = pd.to_datetime(df["date_val"])
    
    cols_meta = _classify_columns_heuristic(df)
    
    assert cols_meta["id_col"]["is_identifier"] is True or cols_meta["id_col"]["semantic_type"] == "numeric"
    assert cols_meta["churn"]["is_target_candidate"] is True
    assert cols_meta["numeric_val"]["semantic_type"] == "numeric"
    assert cols_meta["date_val"]["semantic_type"] == "date"

@patch("data.intelligence.query_ollama")
def test_build_dataset_intelligence(mock_query):
    # Mock LLM response
    mock_llm_json = {
        "dataset_type": "customer_data",
        "domain": "telecom",
        "columns": {
            "id_col": {
                "semantic_type": "identifier",
                "confidence": 0.95,
                "description": "Customer identifier ID"
            },
            "churn": {
                "semantic_type": "binary_target",
                "confidence": 0.90,
                "description": "Indicates if customer churned"
            }
        },
        "hypotheses": [
            "Older customers have a different churn rate"
        ]
    }
    mock_query.return_value = json.dumps(mock_llm_json)
    
    df = pd.DataFrame({
        "id_col": [1, 2, 3],
        "churn": [0, 1, 0]
    })
    
    intel = build_dataset_intelligence(df, "mock.csv", use_cache=False)
    
    assert intel["dataset_type"] == "customer_data"
    assert intel["domain"] == "telecom"
    assert intel["columns"]["churn"]["semantic_type"] == "binary_target"
    assert intel["columns"]["churn"]["is_target_candidate"] is True
    assert intel["columns"]["id_col"]["is_identifier"] is True
    assert len(intel["hypotheses"]) == 1
    assert intel["hypotheses"][0] == "Older customers have a different churn rate"
