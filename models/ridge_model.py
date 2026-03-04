"""Ridge regression model trainer with purged walk-forward CV."""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cv import PurgedWalkForwardCV
from features import MAX_WEIGHT


def train_ridge_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    factor_names: list[str],
    purge_days: int = 2,
    embargo_days: int = 1,
) -> dict:
    """Train Ridge regression with purged walk-forward CV.
    
    Uses adaptive alpha: lower regularization for small feature sets
    to allow more differentiation from equal-weight baseline.
    """
    n_splits = min(5, max(2, len(X) // 15))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)

    # Adaptive alpha: less shrinkage when we have more features and data
    n_features = X.shape[1]
    n_samples = X.shape[0]
    if n_features >= 8 and n_samples >= 20:
        alpha = 0.3   # Enriched feature set — allow differentiation
    elif n_features >= 5:
        alpha = 0.5   # Moderate features
    else:
        alpha = 1.0   # Sparse — regularize more

    cv_scores = []
    cv_correlations = []
    best_model = None
    best_scaler = None
    best_mse = float("inf")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_val_scaled = scaler.transform(X[val_idx])

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y[train_idx])
        preds = model.predict(X_val_scaled)

        mse = float(mean_squared_error(y[val_idx], preds))
        cv_scores.append(mse)

        if len(preds) > 2:
            corr = float(np.corrcoef(preds, y[val_idx])[0, 1])
            if not np.isnan(corr):
                cv_correlations.append(corr)

        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_scaler = scaler

    if best_model is None:
        return {"status": "insufficient_data", "reason": "no_valid_cv_folds"}

    X_scaled = best_scaler.transform(X)
    full_preds = best_model.predict(X_scaled)
    correlation = float(np.corrcoef(full_preds, y)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    direction_correct = np.sum(np.sign(full_preds) == np.sign(y))
    direction_accuracy = float(direction_correct / len(y)) if len(y) > 0 else 0

    raw_coefs = best_model.coef_ / best_scaler.scale_
    coef_weights = {}
    for i, fname in enumerate(factor_names):
        w = float(raw_coefs[i])
        w = np.clip(w, -MAX_WEIGHT, MAX_WEIGHT)
        coef_weights[fname] = round(w, 4)

    abs_coefs = np.abs(raw_coefs)
    total = abs_coefs.sum()
    importance = {}
    for i, fname in enumerate(factor_names):
        importance[fname] = round(float(abs_coefs[i] / total * 100) if total > 0 else 0, 2)

    return {
        "status": "trained",
        "model_type": "ridge",
        "correlation": round(correlation, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "cv_mse": [round(s, 6) for s in cv_scores],
        "cv_correlations": [round(c, 4) for c in cv_correlations],
        "weights": coef_weights,
        "importance": importance,
        "observations": len(X),
        "factors": factor_names,
        "ridge_alpha": alpha,
        "purged_samples": cv.purged_samples,
    }

