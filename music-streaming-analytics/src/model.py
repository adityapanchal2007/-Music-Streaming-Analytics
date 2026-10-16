"""
src/model.py
────────────
Machine learning model training, evaluation, and explainability helpers.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Linear Regression" : lambda: LinearRegression(),
    "Ridge Regression"  : lambda: Ridge(alpha=1.0, random_state=42),
    "Lasso Regression"  : lambda: Lasso(alpha=0.01, random_state=42, max_iter=2000),
    "Decision Tree"     : lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest"     : lambda: RandomForestRegressor(n_estimators=100, max_depth=10,
                                                         n_jobs=-1, random_state=42),
    "Gradient Boosting" : lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                              learning_rate=0.1, random_state=42),
}


def get_model(model_name: str):
    """Instantiate a model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict: MAE, MSE, RMSE, R2
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    return {
        "MAE" : round(mae, 4),
        "MSE" : round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2"  : round(r2, 4),
    }


def train_and_evaluate(model_name: str,
                        X_train_sc: np.ndarray,
                        X_test_sc: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray) -> tuple:
    """
    Train a single model and return (trained_model, y_pred, metrics_dict).
    """
    model = get_model(model_name)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    metrics = evaluate(y_test, y_pred)
    return model, y_pred, metrics


def compare_models(model_names: list,
                   X_train_sc: np.ndarray,
                   X_test_sc: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray) -> pd.DataFrame:
    """
    Train and evaluate multiple models. Returns a comparison DataFrame.

    Columns: Model, MAE, MSE, RMSE, R2
    """
    rows = []
    for name in model_names:
        _, _, metrics = train_and_evaluate(name, X_train_sc, X_test_sc, y_train, y_test)
        rows.append({"Model": name, **metrics})
    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return df


def feature_importance(model, feature_names: list) -> pd.Series:
    """
    Extract feature importances or absolute coefficients from a trained model.

    Returns
    -------
    pd.Series sorted by importance (descending). None if not available.
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_,
                         index=feature_names).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_),
                         index=feature_names).sort_values(ascending=False)
    return None


def prediction_table(y_test: np.ndarray, y_pred: np.ndarray,
                      n: int = 20) -> pd.DataFrame:
    """
    Build a sample prediction vs. actual table.

    Returns
    -------
    pd.DataFrame with columns:
        Actual_log_streams, Predicted_log_streams, Error,
        Actual_streams_approx, Predicted_streams_approx
    """
    idx = np.random.default_rng(42).choice(len(y_test), size=min(n, len(y_test)),
                                            replace=False)
    return pd.DataFrame({
        "Actual_log_streams"      : y_test[idx].round(4),
        "Predicted_log_streams"   : y_pred[idx].round(4),
        "Error"                   : (y_test[idx] - y_pred[idx]).round(4),
        "Actual_streams_approx"   : np.expm1(y_test[idx]).astype(int),
        "Predicted_streams_approx": np.expm1(y_pred[idx]).astype(int),
    })
