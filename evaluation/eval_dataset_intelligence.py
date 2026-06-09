import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
from unittest.mock import patch
from data.intelligence import build_dataset_intelligence

EXPECTED = {
    "titanic.csv": {
        "id_cols": ["PassengerId"],
        "target_cols": ["Survived"],
        "date_cols": [],
        "semantic_types": {
            "PassengerId": "identifier",
            "Survived": "binary_target",
            "Pclass": "categorical",
            "Sex": "categorical",
            "Age": "numeric",
            "Fare": "numeric"
        }
    },
    "customer_churn.csv": {
        "id_cols": ["CustomerID"],
        "target_cols": ["Churn"],
        "date_cols": [],
        "semantic_types": {
            "CustomerID": "identifier",
            "Churn": "binary_target",
            "Tenure": "numeric",
            "MonthlyCharges": "numeric"
        }
    },
    "sales_transactions.csv": {
        "id_cols": ["TransactionID"],
        "target_cols": ["Revenue"],
        "date_cols": ["Date"],
        "semantic_types": {
            "TransactionID": "identifier",
            "Revenue": "numeric",
            "Quantity": "numeric",
            "UnitPrice": "numeric",
            "Date": "date"
        }
    },
    "credit_risk.csv": {
        "id_cols": ["LoanID"],
        "target_cols": ["Default"],
        "date_cols": [],
        "semantic_types": {
            "LoanID": "identifier",
            "Default": "binary_target",
            "Age": "numeric",
            "Income": "numeric",
            "LoanAmount": "numeric",
            "CreditScore": "numeric"
        }
    },
    "employee_attrition.csv": {
        "id_cols": ["EmployeeID"],
        "target_cols": ["Attrition"],
        "date_cols": [],
        "semantic_types": {
            "EmployeeID": "identifier",
            "Attrition": "binary_target",
            "Age": "numeric",
            "MonthlyIncome": "numeric",
            "YearsAtCompany": "numeric",
            "WorkLifeBalance": "categorical"
        }
    },
    "patient_treatment.csv": {
        "id_cols": ["PatientID"],
        "target_cols": ["Outcome"],
        "date_cols": [],
        "semantic_types": {
            "PatientID": "identifier",
            "Outcome": "binary_target",
            "Age": "numeric",
            "Diagnosis": "categorical",
            "Treatment": "categorical"
        }
    }
}

def mock_query_ollama(model, prompt, system="", timeout=300):
    if "titanic.csv" in prompt:
        return json.dumps({
            "dataset_type": "passenger_survival",
            "domain": "maritime",
            "columns": {
                "PassengerId": {"semantic_type": "identifier", "confidence": 0.9, "description": "Unique ID"},
                "Survived": {"semantic_type": "binary_target", "confidence": 0.9, "description": "Survival flag"},
                "Pclass": {"semantic_type": "categorical", "confidence": 0.8, "description": "Ticket class"},
                "Sex": {"semantic_type": "categorical", "confidence": 0.9, "description": "Gender"},
                "Age": {"semantic_type": "numeric", "confidence": 0.9, "description": "Age in years"},
                "Fare": {"semantic_type": "numeric", "confidence": 0.9, "description": "Passenger fare"}
            },
            "hypotheses": ["Higher class passengers had higher survival rates"]
        })
    elif "customer_churn.csv" in prompt:
        return json.dumps({
            "dataset_type": "customer_churn",
            "domain": "telecom",
            "columns": {
                "CustomerID": {"semantic_type": "identifier", "confidence": 0.9, "description": "Customer ID"},
                "Churn": {"semantic_type": "binary_target", "confidence": 0.9, "description": "Churn indicator"},
                "Tenure": {"semantic_type": "numeric", "confidence": 0.9, "description": "Months of tenure"},
                "MonthlyCharges": {"semantic_type": "numeric", "confidence": 0.9, "description": "Monthly charge"}
            },
            "hypotheses": ["Longer tenure customers churn less"]
        })
    elif "sales_transactions.csv" in prompt:
        return json.dumps({
            "dataset_type": "sales_transactions",
            "domain": "retail",
            "columns": {
                "TransactionID": {"semantic_type": "identifier", "confidence": 0.9, "description": "Transaction ID"},
                "Revenue": {"semantic_type": "numeric", "confidence": 0.9, "description": "Transaction revenue"},
                "Quantity": {"semantic_type": "numeric", "confidence": 0.9, "description": "Quantity purchased"},
                "UnitPrice": {"semantic_type": "numeric", "confidence": 0.9, "description": "Unit price"},
                "Date": {"semantic_type": "date", "confidence": 0.9, "description": "Transaction date"}
            },
            "hypotheses": ["Revenue peaks during weekends"]
        })
    elif "credit_risk.csv" in prompt:
        return json.dumps({
            "dataset_type": "credit_risk",
            "domain": "finance",
            "columns": {
                "LoanID": {"semantic_type": "identifier", "confidence": 0.9, "description": "Loan ID"},
                "Default": {"semantic_type": "binary_target", "confidence": 0.9, "description": "Default status"},
                "Age": {"semantic_type": "numeric", "confidence": 0.9, "description": "Borrower age"},
                "Income": {"semantic_type": "numeric", "confidence": 0.9, "description": "Annual income"},
                "LoanAmount": {"semantic_type": "numeric", "confidence": 0.9, "description": "Requested loan amount"},
                "CreditScore": {"semantic_type": "numeric", "confidence": 0.9, "description": "Credit score"}
            },
            "hypotheses": ["Lower credit scores are linked to higher default rates"]
        })
    elif "employee_attrition.csv" in prompt:
        return json.dumps({
            "dataset_type": "employee_attrition",
            "domain": "hr",
            "columns": {
                "EmployeeID": {"semantic_type": "identifier", "confidence": 0.9, "description": "Employee ID"},
                "Attrition": {"semantic_type": "binary_target", "confidence": 0.9, "description": "Attrition status"},
                "Age": {"semantic_type": "numeric", "confidence": 0.9, "description": "Employee age"},
                "MonthlyIncome": {"semantic_type": "numeric", "confidence": 0.9, "description": "Monthly income"},
                "YearsAtCompany": {"semantic_type": "numeric", "confidence": 0.9, "description": "Years at the company"},
                "WorkLifeBalance": {"semantic_type": "categorical", "confidence": 0.9, "description": "Work life balance rating"}
            },
            "hypotheses": ["Employees with lower monthly income have higher attrition rates."]
        })
    elif "patient_treatment.csv" in prompt:
        return json.dumps({
            "dataset_type": "patient_treatment",
            "domain": "healthcare",
            "columns": {
                "PatientID": {"semantic_type": "identifier", "confidence": 0.9, "description": "Patient ID"},
                "Outcome": {"semantic_type": "binary_target", "confidence": 0.9, "description": "Treatment outcome"},
                "Age": {"semantic_type": "numeric", "confidence": 0.9, "description": "Patient age"},
                "Diagnosis": {"semantic_type": "categorical", "confidence": 0.9, "description": "Diagnosis category"},
                "Treatment": {"semantic_type": "categorical", "confidence": 0.9, "description": "Treatment type"}
            },
            "hypotheses": ["Treatment type A leads to better recovery rates."]
        })
    return "{}"

@patch("data.intelligence.query_ollama", side_effect=mock_query_ollama)
def run_evaluation(mock_query):
    demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo")
    
    total_checks = 0
    passed_checks = 0
    
    target_checks = 0
    target_passed = 0
    
    id_checks = 0
    id_passed = 0
    
    date_checks = 0
    date_passed = 0
    
    type_checks = 0
    type_passed = 0

    print("=" * 60)
    print("  Dataset Intelligence Layer Benchmarking Evaluation")
    print("=" * 60)
    print(f"{'Dataset':<22} | {'Targets':<8} | {'IDs':<8} | {'Dates':<8} | {'Types':<8}")
    print("-" * 60)

    failures = []

    for filename, expected in EXPECTED.items():
        filepath = os.path.join(demo_dir, filename)
        if not os.path.exists(filepath):
            print(f"File missing: {filename}")
            continue
            
        df = pd.read_csv(filepath)
        intel = build_dataset_intelligence(df, filename, use_cache=False)
        
        # 1. Target checks
        target_checks += 1
        target_detected = intel.get("target_candidates", [])
        target_ok = any(t in target_detected for t in expected["target_cols"])
        if target_ok:
            target_passed += 1
            passed_checks += 1
        else:
            failures.append({
                "dataset": filename,
                "column": expected["target_cols"][0] if expected["target_cols"] else "N/A",
                "expected": "target",
                "predicted": "not_detected"
            })
        total_checks += 1
            
        # 2. ID checks
        id_checks += 1
        id_detected = intel.get("id_columns", [])
        id_ok = any(i in id_detected for i in expected["id_cols"])
        if id_ok:
            id_passed += 1
            passed_checks += 1
        else:
            failures.append({
                "dataset": filename,
                "column": expected["id_cols"][0] if expected["id_cols"] else "N/A",
                "expected": "identifier",
                "predicted": "not_detected"
            })
        total_checks += 1

        # 3. Date checks
        date_checks += 1
        date_detected = intel.get("date_columns", [])
        date_ok = set(date_detected) == set(expected["date_cols"])
        if date_ok:
            date_passed += 1
            passed_checks += 1
        else:
            failures.append({
                "dataset": filename,
                "column": expected["date_cols"][0] if expected["date_cols"] else "N/A",
                "expected": "date",
                "predicted": "mismatch"
            })
        total_checks += 1
        
        # 4. Semantic types
        cols = intel.get("columns", {})
        ds_type_passed = 0
        ds_type_checks = 0
        for col_name, expected_type in expected["semantic_types"].items():
            if col_name in cols:
                ds_type_checks += 1
                type_checks += 1
                detected_type = cols[col_name].get("semantic_type")
                match = (detected_type == expected_type or
                         (expected_type == "numeric" and detected_type in ("numeric", "currency", "percentage")) or
                         (expected_type == "binary_target" and detected_type in ("binary_target", "target")))
                if match:
                    ds_type_passed += 1
                    type_passed += 1
                    passed_checks += 1
                else:
                    failures.append({
                        "dataset": filename,
                        "column": col_name,
                        "expected": expected_type,
                        "predicted": detected_type
                    })
                total_checks += 1

        t_status = "PASS" if target_ok else "FAIL"
        i_status = "PASS" if id_ok else "FAIL"
        d_status = "PASS" if date_ok else "FAIL"
        type_pct = (ds_type_passed / ds_type_checks * 100) if ds_type_checks > 0 else 0
        
        print(f"{filename:<22} | {t_status:<8} | {i_status:<8} | {d_status:<8} | {type_pct:>6.1f}%")

    print("-" * 60)
    overall_acc = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    t_acc = (target_passed / target_checks * 100) if target_checks > 0 else 0
    i_acc = (id_passed / id_checks * 100) if id_checks > 0 else 0
    d_acc = (date_passed / date_checks * 100) if date_checks > 0 else 0
    sem_acc = (type_passed / type_checks * 100) if type_checks > 0 else 0

    print(f"Target Detection Accuracy:      {t_acc:.1f}%")
    print(f"Identifier Detection Accuracy:  {i_acc:.1f}%")
    print(f"Date Column Detection Accuracy: {d_acc:.1f}%")
    print(f"Semantic Typing Accuracy:       {sem_acc:.1f}%")
    print(f"Overall Accuracy:               {overall_acc:.1f}%")
    print("=" * 60)

    # Write failure report
    failures_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intelligence_failures.json")
    with open(failures_file, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"Wrote intelligence failures report to {failures_file}")

if __name__ == "__main__":
    run_evaluation()
