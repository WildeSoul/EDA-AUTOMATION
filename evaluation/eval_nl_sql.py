import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from unittest.mock import patch
from data.engine import DuckDBEngine
from data.nl_sql import execute_nl_query
from data.intelligence import build_dataset_intelligence

BENCHMARKS = [
    # Titanic
    {
        "dataset": "titanic.csv",
        "question": "What is the average fare by passenger class?",
        "expected_columns": ["Pclass", "avg_fare"],
        "check_fn": lambda df: len(df) == 3 and "Pclass" in df.columns
    },
    {
        "dataset": "titanic.csv",
        "question": "Count of survivors group by sex",
        "expected_columns": ["Sex", "count"],
        "check_fn": lambda df: len(df) == 2 and "Sex" in df.columns
    },
    {
        "dataset": "titanic.csv",
        "question": "List name and age of oldest passengers who survived",
        "expected_columns": ["Name", "Age"],
        "check_fn": lambda df: len(df) > 0 and "Age" in df.columns
    },
    
    # Customer Churn
    {
        "dataset": "customer_churn.csv",
        "question": "What is the total monthly charge for churned customers vs non-churned?",
        "expected_columns": ["Churn", "total"],
        "check_fn": lambda df: len(df) == 2 and "Churn" in df.columns
    },
    {
        "dataset": "customer_churn.csv",
        "question": "Show average tenure by contract type",
        "expected_columns": ["Contract", "avg_tenure"],
        "check_fn": lambda df: len(df) == 3 and "Contract" in df.columns
    },
    
    # Sales Transactions
    {
        "dataset": "sales_transactions.csv",
        "question": "What is the total revenue by product category?",
        "expected_columns": ["ProductCategory", "total_revenue"],
        "check_fn": lambda df: len(df) > 0 and "ProductCategory" in df.columns
    },
    {
        "dataset": "sales_transactions.csv",
        "question": "Find the transaction with highest quantity sold",
        "expected_columns": ["TransactionID", "Quantity"],
        "check_fn": lambda df: len(df) > 0 and "Quantity" in df.columns
    },
    
    # Credit Risk
    {
        "dataset": "credit_risk.csv",
        "question": "Show average credit score and income by default status",
        "expected_columns": ["Default", "CreditScore", "Income"],
        "check_fn": lambda df: len(df) == 2 and "Default" in df.columns
    },
    {
        "dataset": "credit_risk.csv",
        "question": "Count of loans where age is under 30 and default is 1",
        "expected_columns": ["count"],
        "check_fn": lambda df: len(df) == 1
    },
    # Employee Attrition
    {
        "dataset": "employee_attrition.csv",
        "question": "What is the average monthly income for employees who attrited vs those who stayed?",
        "expected_columns": ["Attrition", "avg_income"],
        "check_fn": lambda df: len(df) == 2 and "Attrition" in df.columns
    },
    {
        "dataset": "employee_attrition.csv",
        "question": "Show count of employees by work life balance score",
        "expected_columns": ["WorkLifeBalance", "count"],
        "check_fn": lambda df: len(df) > 0 and "WorkLifeBalance" in df.columns
    },
    # Patient Treatment
    {
        "dataset": "patient_treatment.csv",
        "question": "What is the average age by diagnosis group?",
        "expected_columns": ["Diagnosis", "avg_age"],
        "check_fn": lambda df: len(df) > 0 and "Diagnosis" in df.columns
    },
    {
        "dataset": "patient_treatment.csv",
        "question": "List diagnosis and outcome counts for patients",
        "expected_columns": ["Diagnosis", "Outcome", "count"],
        "check_fn": lambda df: len(df) > 0 and "Diagnosis" in df.columns and "Outcome" in df.columns
    }
]

def mock_query_ollama(model, prompt, system="", timeout=300):
    if "average fare by passenger class" in prompt:
        return "SELECT Pclass, AVG(Fare) as avg_fare FROM dataset GROUP BY Pclass ORDER BY Pclass;\nExplanation: Average passenger fare grouped by passenger class."
    elif "Count of survivors group by sex" in prompt:
        return "SELECT Sex, COUNT(*) as count FROM dataset WHERE Survived = 1 GROUP BY Sex;\nExplanation: Survivor counts for each gender."
    elif "List name and age of oldest passengers who survived" in prompt:
        return "SELECT Name, Age FROM dataset WHERE Survived = 1 ORDER BY Age DESC LIMIT 5;\nExplanation: Oldest survivors."
    elif "total monthly charge for churned customers vs non-churned" in prompt:
        return "SELECT Churn, SUM(MonthlyCharges) as total FROM dataset GROUP BY Churn;\nExplanation: Sum of monthly charges grouped by churn status."
    elif "Show average tenure by contract type" in prompt:
        return "SELECT Contract, AVG(Tenure) as avg_tenure FROM dataset GROUP BY Contract;\nExplanation: Average customer tenure grouped by contract type."
    elif "total revenue by product category" in prompt:
        return "SELECT ProductCategory, SUM(Revenue) as total_revenue FROM dataset GROUP BY ProductCategory;\nExplanation: Sum of revenue by category."
    elif "Find the transaction with highest quantity sold" in prompt:
        return "SELECT TransactionID, Quantity FROM dataset ORDER BY Quantity DESC LIMIT 1;\nExplanation: Transaction with maximum quantity."
    elif "average credit score and income by default status" in prompt:
        return 'SELECT "Default", AVG(CreditScore) as CreditScore, AVG(Income) as Income FROM dataset GROUP BY "Default";\nExplanation: Average credit score and income grouped by loan default status.'
    elif "Count of loans where age is under 30 and default is 1" in prompt:
        return 'SELECT COUNT(*) as count FROM dataset WHERE Age < 30 AND "Default" = 1;\nExplanation: Number of defaulted loans for borrowers under 30.'
    elif "average monthly income for employees who attrited vs those who stayed" in prompt:
        return 'SELECT Attrition, AVG(MonthlyIncome) as avg_income FROM dataset GROUP BY Attrition;\nExplanation: Average monthly income grouped by attrition status.'
    elif "Show count of employees by work life balance score" in prompt:
        return 'SELECT WorkLifeBalance, COUNT(*) as count FROM dataset GROUP BY WorkLifeBalance;\nExplanation: Employee count grouped by work life balance score.'
    elif "average age by diagnosis group" in prompt:
        return 'SELECT Diagnosis, AVG(Age) as avg_age FROM dataset GROUP BY Diagnosis;\nExplanation: Average patient age grouped by diagnosis.'
    elif "List diagnosis and outcome counts for patients" in prompt:
        return 'SELECT Diagnosis, Outcome, COUNT(*) as count FROM dataset GROUP BY Diagnosis, Outcome;\nExplanation: Diagnosis and outcome frequency counts.'
    return "SELECT * FROM dataset LIMIT 10;\nExplanation: Default fallback query."

def mock_query_ollama_intel(model, prompt, system="", timeout=300):
    # Dummy mock for intelligence layer schema queries
    return "{}"

@patch("data.nl_sql.query_ollama", side_effect=mock_query_ollama)
@patch("data.intelligence.query_ollama", side_effect=mock_query_ollama_intel)
def run_evaluation(mock_intel_query, mock_sql_query):
    demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo")
    
    success_count = 0
    sql_generation_success = 0
    sql_execution_success = 0
    
    print("=" * 70)
    print("  NL to SQL Benchmark Evaluation")
    print("=" * 70)
    print(f"{'Question':<50} | {'Gen':<5} | {'Exec':<5} | {'Val':<5}")
    print("-" * 70)
    
    sessions = {}
    
    for item in BENCHMARKS:
        filename = item["dataset"]
        filepath = os.path.join(demo_dir, filename)
        if not os.path.exists(filepath):
            print(f"Dataset missing: {filename}")
            continue
            
        if filename not in sessions:
            df = pd.read_csv(filepath)
            intel = build_dataset_intelligence(df, filename, use_cache=False)
            engine = DuckDBEngine()
            engine.register(df)
            sessions[filename] = (intel, engine)
            
        intel, engine = sessions[filename]
        res = execute_nl_query(item["question"], engine, "dataset", intel)
        
        gen_ok = bool(res.get("sql"))
        exec_ok = res.get("success", False)
        val_ok = False
        
        if gen_ok:
            sql_generation_success += 1
        if exec_ok:
            sql_execution_success += 1
            df_res = res.get("result")
            if df_res is not None:
                try:
                    val_ok = item["check_fn"](df_res)
                except Exception:
                    val_ok = False
                    
        if val_ok:
            success_count += 1
            
        g_status = "OK" if gen_ok else "FAIL"
        e_status = "OK" if exec_ok else "FAIL"
        v_status = "OK" if val_ok else "FAIL"
        
        trunc_q = item["question"][:47] + "..." if len(item["question"]) > 50 else item["question"]
        print(f"{trunc_q:<50} | {g_status:<5} | {e_status:<5} | {v_status:<5}")
        
    print("-" * 70)
    total = len(BENCHMARKS)
    gen_pct = (sql_generation_success / total * 100) if total > 0 else 0
    exec_pct = (sql_execution_success / total * 100) if total > 0 else 0
    val_pct = (success_count / total * 100) if total > 0 else 0
    
    print(f"SQL Generation Success: {gen_pct:.1f}%")
    print(f"SQL Execution Success:  {exec_pct:.1f}%")
    print(f"Validation Pass Rate:   {val_pct:.1f}%")
    print("=" * 70)
    
    for intel, engine in sessions.values():
        engine.close()

if __name__ == "__main__":
    run_evaluation()
