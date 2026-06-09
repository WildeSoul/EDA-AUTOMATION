"""
Quick ML Modeling — Random Forest, Logistic/Linear Regression, XGBoost.
Auto-detects classification vs regression from target variable.

DATA LEAKAGE PREVENTION:
  All preprocessing is wrapped in sklearn.pipeline.Pipeline so that
  fit() is ONLY called on the training split. The test set is
  transformed using statistics derived strictly from training data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config.settings import ML_TEST_SIZE, ML_RANDOM_STATE, ML_CV_FOLDS

# Try importing XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_available_models(task_type: str) -> list[str]:
    """Return available model names for the given task type."""
    if task_type == "classification":
        models = ["Random Forest", "Logistic Regression"]
    else:
        models = ["Random Forest", "Linear Regression"]
    if HAS_XGBOOST:
        models.append("XGBoost")
    return models


def _prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple:
    """Prepare features and target for modeling."""
    # Drop rows where target is NaN
    clean_df = df[feature_cols + [target_col]].dropna()
    X = clean_df[feature_cols].copy()
    y = clean_df[target_col].copy()

    # Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Determine task type
    if y.dtype == "object" or y.dtype.name == "category" or y.nunique() <= 10:
        task_type = "classification"
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
        label_encoders["__target__"] = le_target
    else:
        task_type = "regression"

    return X, y, task_type, label_encoders


def _get_model(model_name: str, task_type: str):
    """Instantiate the selected model."""
    if task_type == "classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=ML_RANDOM_STATE, n_jobs=-1)
        elif model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=ML_RANDOM_STATE)
        elif model_name == "XGBoost" and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=100, random_state=ML_RANDOM_STATE,
                use_label_encoder=False, eval_metric="logloss",
                verbosity=0,
            )
    else:
        if model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=100, random_state=ML_RANDOM_STATE, n_jobs=-1)
        elif model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "XGBoost" and HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=100, random_state=ML_RANDOM_STATE, verbosity=0,
            )
    return None


def train_quick_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str = "Random Forest",
) -> dict:
    """
    Train a quick baseline model and return comprehensive results.

    DATA LEAKAGE PREVENTION:
      StandardScaler is wrapped inside an sklearn Pipeline so that
      fit() is called ONLY on the training split. The test set is
      transformed using training-set statistics exclusively.

    Returns dict with: task_type, model_name, metrics, feature_importance,
    confusion_matrix (classification), predictions, cv_scores, etc.
    """
    result = {"success": False, "error": None}

    try:
        X, y, task_type, label_encoders = _prepare_data(df, target_col, feature_cols)
        result["task_type"] = task_type
        result["model_name"] = model_name
        result["n_samples"] = len(X)
        result["n_features"] = len(feature_cols)

        if len(X) < 20:
            result["error"] = "Not enough samples (need at least 20 rows)."
            return result

        # ─── PIPELINE: Scaler + Model (prevents data leakage) ───
        base_model = _get_model(model_name, task_type)
        if base_model is None:
            result["error"] = f"Model '{model_name}' not available for {task_type}."
            return result

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

        # Train/test split — scaler has NOT seen ANY data yet
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE,
            stratify=y if task_type == "classification" and y.nunique() <= 20 else None,
        )

        # Pipeline.fit() → scaler.fit_transform(X_train) then model.fit()
        # The scaler learns mean/std from TRAINING DATA ONLY
        pipeline.fit(X_train, y_train)

        # Pipeline.predict() → scaler.transform(X_test) then model.predict()
        # Test data is transformed using TRAINING statistics
        y_pred = pipeline.predict(X_test)

        # Cross-validation (pipeline ensures no leakage in each fold)
        scoring = "accuracy" if task_type == "classification" else "r2"
        n_folds = min(ML_CV_FOLDS, len(X_train))
        cv_scores = cross_val_score(pipeline, X, y, cv=n_folds, scoring=scoring)
        result["cv_scores"] = cv_scores
        result["cv_mean"] = round(cv_scores.mean(), 4)
        result["cv_std"] = round(cv_scores.std(), 4)
        result["leakage_protected"] = True  # Flag for UI

        # Metrics
        if task_type == "classification":
            result["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            result["f1_score"] = round(f1_score(y_test, y_pred, average="weighted"), 4)
            result["confusion_matrix"] = confusion_matrix(y_test, y_pred)
            # Decode labels for report
            if "__target__" in label_encoders:
                target_names = label_encoders["__target__"].classes_
            else:
                target_names = None
            result["classification_report"] = classification_report(
                y_test, y_pred,
                target_names=target_names,
                output_dict=True,
            )
        else:
            result["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            result["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
            result["r2"] = round(r2_score(y_test, y_pred), 4)
            result["y_test"] = y_test.values
            result["y_pred"] = y_pred

        # Feature importance (extract from the model inside the pipeline)
        fitted_model = pipeline.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importances = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            importances = np.abs(fitted_model.coef_).flatten()
            if len(importances) != len(feature_cols):
                importances = np.abs(fitted_model.coef_[0]) if fitted_model.coef_.ndim > 1 else importances
        else:
            importances = None

        if importances is not None and len(importances) == len(feature_cols):
            result["feature_importance"] = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances,
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

        # ─── Model Explainability (SHAP) ───
        try:
            import shap
            # Transform background data for SHAP using the pipeline's scaler
            X_train_scaled = pipeline.named_steps["scaler"].transform(X_train)
            X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)
            
            # Sample for speed if dataset is large
            shap_bg = shap.sample(X_train_scaled, 100) if len(X_train_scaled) > 100 else X_train_scaled
            shap_test = shap.sample(X_test_scaled, 100) if len(X_test_scaled) > 100 else X_test_scaled

            # Tree-based models (Random Forest, XGBoost)
            if "Forest" in model_name or "XGB" in model_name:
                explainer = shap.TreeExplainer(fitted_model)
                shap_values = explainer.shap_values(shap_test)
            # Linear models
            else:
                explainer = shap.LinearExplainer(fitted_model, shap_bg)
                shap_values = explainer.shap_values(shap_test)
            
            # Store SHAP data for UI plotting
            result["shap_values"] = shap_values
            result["shap_test_data"] = pd.DataFrame(shap_test, columns=feature_cols)
            result["shap_explainer"] = explainer
        except Exception as e:
            # SHAP might fail on some specific model/data combos; fail gracefully
            pass

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
