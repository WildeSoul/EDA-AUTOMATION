import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.schema_agent import SchemaAgent
from agents.quality_agent import QualityAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.eda_agent import EDAAgent
from agents.insight_agent import InsightAgent
from agents.router import route_request, run_agent_pipeline

@pytest.fixture
def sample_intelligence():
    return {
        "filename": "data.csv",
        "dataset_fingerprint": "mock_fp_123",
        "dataset_type": "customer_churn",
        "domain": "telecom",
        "row_count": 100,
        "column_count": 3,
        "columns": {
            "CustomerID": {
                "dtype": "int64",
                "semantic_type": "identifier",
                "confidence": 0.8,
                "nunique": 100,
                "missing_pct": 0.0,
                "sample_values": [1, 2, 3],
                "is_identifier": True,
                "is_target_candidate": False
            },
            "Tenure": {
                "dtype": "int64",
                "semantic_type": "numeric",
                "confidence": 0.9,
                "nunique": 20,
                "missing_pct": 0.0,
                "sample_values": [5, 10, 15],
                "is_identifier": False,
                "is_target_candidate": False
            },
            "Churn": {
                "dtype": "int64",
                "semantic_type": "binary_target",
                "confidence": 0.9,
                "nunique": 2,
                "missing_pct": 0.0,
                "sample_values": [0, 1],
                "is_identifier": False,
                "is_target_candidate": True
            }
        },
        "target_candidates": ["Churn"],
        "id_columns": ["CustomerID"],
        "numeric_columns": ["Tenure"],
        "categorical_columns": ["Churn"],
        "quality_score": 95,
        "quality_grade": "A",
        "quality_issues": [],
        "hypotheses": ["Tenure is related to Churn"]
    }

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "CustomerID": list(range(100)),
        "Tenure": [x % 20 for x in range(100)],
        "Churn": [x % 2 for x in range(100)]
    })

@patch("agents.schema_agent.query_ollama")
def test_schema_agent(mock_query, sample_intelligence):
    mock_response = {
        "relationships": [{"columns": ["Tenure", "Churn"], "type": "correlated", "description": "Tenure correlates with churn"}],
        "modeling_advice": [],
        "recommended_target": "Churn",
        "recommended_target_reason": "It is binary target",
        "data_type_fixes": [],
        "summary": "Good schema readiness"
    }
    mock_query.return_value = json.dumps(mock_response)
    
    agent = SchemaAgent()
    res = agent.run(sample_intelligence)
    
    assert res["success"] is True
    assert res["result"]["recommended_target"] == "Churn"
    assert "correlated: Tenure ↔ Churn" in res["evidence"][0]

@patch("agents.quality_agent.query_ollama")
def test_quality_agent(mock_query, sample_intelligence, sample_df):
    mock_query.return_value = "This dataset is in good health."
    
    agent = QualityAgent()
    res = agent.run(sample_intelligence, df=sample_df)
    
    assert res["success"] is True
    assert res["result"]["overall_assessment"] == "excellent"
    assert "passes all quality checks" in res["result"]["narrative"] or "health" in res["result"]["narrative"]

def test_router_routing(sample_intelligence):
    agents = route_request("upload", sample_intelligence)
    assert agents == ["schema", "quality"]
    
    agents_full = route_request("full_analysis", sample_intelligence)
    assert agents_full == ["schema", "quality", "hypothesis", "eda", "insight"]
    
    agents_chat = route_request("chat", sample_intelligence, query="check missing values and clean data")
    assert "quality" in agents_chat
