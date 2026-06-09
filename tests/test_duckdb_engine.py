import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from data.engine import DuckDBEngine

def test_duckdb_engine_registration_and_query():
    engine = DuckDBEngine()
    
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000, 60000, 70000]
    })
    
    meta = engine.register(df, "employees")
    assert meta["table_name"] == "employees"
    assert meta["row_count"] == 3
    assert "age" in meta["columns"]
    
    schema = engine.get_schema("employees")
    assert any(col.get("column_name") == "age" or col.get("Field") == "age" for col in schema)
    
    # Simple query
    res = engine.query("SELECT AVG(age) as avg_age FROM employees")
    assert res.shape == (1, 1)
    assert float(res["avg_age"].iloc[0]) == 30.0
    
    # Query with filter
    res_filtered = engine.query("SELECT name FROM employees WHERE age > 28")
    assert len(res_filtered) == 2
    assert list(res_filtered["name"]) == ["Bob", "Charlie"]
    
    engine.close()

def test_duckdb_engine_dangerous_queries():
    engine = DuckDBEngine()
    df = pd.DataFrame({"x": [1, 2]})
    engine.register(df, "test_table")
    
    # UPDATE should fail
    with pytest.raises(ValueError) as excinfo:
        engine.query("UPDATE test_table SET x = 3")
    assert "Write operations" in str(excinfo.value)
    
    # DROP should fail
    with pytest.raises(ValueError) as excinfo:
        engine.query("DROP TABLE test_table")
    assert "Write operations" in str(excinfo.value)
    
    engine.close()
