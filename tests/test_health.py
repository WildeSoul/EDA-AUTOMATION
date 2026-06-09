import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.health import compute_data_health_score

def test_health_score_perfect_data():
    """Test that perfect data receives a score of 100."""
    # Create perfect data (no missing, no dups, no mixed types, normal distribution)
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric1': np.random.normal(loc=0, scale=1, size=1000),
        'numeric2': np.random.normal(loc=5, scale=2, size=1000),
        'categorical1': np.random.choice(['A', 'B', 'C'], size=1000)
    })
    
    health = compute_data_health_score(df, ['numeric1', 'numeric2'], ['categorical1'])
    
    assert health['total_score'] <= 100, "Score should never exceed 100"
    assert health['total_score'] >= 90, "Perfect data should score at least 90 (Grade A)"
    assert health['grade'] == 'A'

def test_health_score_missing_data():
    """Test that missing data lowers the completeness score."""
    df = pd.DataFrame({
        'numeric1': [1, 2, np.nan, 4, 5],
        'numeric2': [np.nan, np.nan, np.nan, 4, 5], # 60% missing
    })
    
    health = compute_data_health_score(df, ['numeric1', 'numeric2'], [])
    
    # 4/10 cells missing = 60% complete. 0.6 * 25 max points = 15 points
    completeness_score = health['breakdown']['Completeness']['score']
    assert completeness_score == 15.0

def test_health_score_high_cardinality():
    """Test that ID-like columns lower the uniqueness score."""
    df = pd.DataFrame({
        'id_col': [f'ID_{i}' for i in range(100)], # 100 unique string values in 100 rows
        'target': [1, 0] * 50
    })
    
    health = compute_data_health_score(df, ['target'], ['id_col'])
    
    # High cardinality penalty is -2 points from the 10 max cardinality points
    uniqueness_score = health['breakdown']['Uniqueness']['score']
    assert uniqueness_score == 18.0 # 10 for no dups + 8 for cardinality

def test_health_score_never_exceeds_100():
    """Test to guarantee score never exceeds 100 even with edge cases."""
    df = pd.DataFrame({
        'a': [1],
        'b': [2]
    })
    health = compute_data_health_score(df, ['a', 'b'], [])
    assert 0 <= health['total_score'] <= 100
