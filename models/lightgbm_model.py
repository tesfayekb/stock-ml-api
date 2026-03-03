
```python
"""LightGBM model trainer with purged walk-forward CV and SHAP."""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from cv import PurgedWalkForwardCV
from features import compute_shap_importance, MAX_WEIGHT


def train_lightgbm_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    factor_names: list[str],
    purge_days: int = 2,
    embargo_days: int = 1,
    user_id: str = "",
    ticker: str = "",
) -> dict:
    """Train LightGBM with purged walk-forward CV."""
    n_splits = min(5, max(2, len(X) // 15))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
    cv_scores = []
    cv_correlations = []
    best_model = None
    best_mse = float("inf")
    best_val_idx = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(train_idx) // 20),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        mse = float(mean_squared_error(y[val_idx], preds))
        cv_scores.append(mse)

        if len(preds) > 2:
            corr = float(np.corrcoef(preds, y[val_idx])[0, 1])
            if not np.isnan(corr):
                cv_correlations.append(corr)

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_val_idx = val_idx

    if best_model is None:
        return {"status": "insufficient_data", "reason": "no_valid_cv_folds"}

    full_preds = best_model.predict(X)
    correlation = float(np.corrcoef(full_preds, y)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    importances = best_model.feature_importances_
    importance_dict = {
        factor_names[i]: int(importances[i])
        for i in range(len(factor_names))
    }

    direction_correct = np.sum(np.sign(full_preds) == np.sign(y))
    direction_accuracy = float(direction_correct / len(y)) if len(y) > 0 else 0

    shap_importance = {}
    if best_val_idx is not None and len(best_val_idx) >= 3:
        shap_importance = compute_shap_importance(
            best_model, X[best_val_idx], factor_names
        )

    weights = {}
    total_imp = sum(importances)
    if total_imp > 0:
        normed = importances / total_imp
        for i, factor in enumerate(factor_names):
            raw_w = 1.0 + normed[i] * (MAX_WEIGHT - 1.0) * 2
            weights[factor] = round(float(np.clip(raw_w, -MAX_WEIGHT, MAX_WEIGHT)), 4)

    return {
        "status": "trained",
        "model_type": "lightgbm",
        "correlation": round(correlation, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "cv_mse": [round(s, 6) for s in cv_scores],
        "cv_correlations": [round(c, 4) for c in cv_correlations],
        "weights": weights,
        "importance": importance_dict,
        "shap_importance": shap_importance,
        "observations": len(X),
        "factors": factor_names,
        "purged_samples": cv.purged_samples,
    }
