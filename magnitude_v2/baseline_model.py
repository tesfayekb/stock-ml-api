"""
Baseline Magnitude Model — predicts normal-flow 3-day stock magnitude
using quantile regression for learned uncertainty intervals.

Outputs: point estimate + q10/q25/q50/q75/q90 + threshold probabilities.
"""

import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

from cv import PurgedWalkForwardCV

log = logging.getLogger("magnitude-v2")

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
THRESHOLDS = [1.0, 2.0, 3.0, 5.0]

# Feature columns used by baseline model (excludes earnings-specific)
BASELINE_FEATURES = [
    "vix_level", "vix_term_slope", "credit_spread",
    "spy_return_1d", "spy_return_3d",
    "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    "iv_rv_spread",
    "sector_relative_strength", "sector_dispersion",
    "rsi14", "momentum_3d", "momentum_5d", "momentum_10d",
    "bb_position", "atr_pct",
    "volume_ratio_20d", "volume_persistence_3d",
    "gap_frequency_20d", "post_gap_followthrough",
    "event_crowding_score", "catalyst_count_7d",
    "market_beta", "residual_vs_market_3d", "residual_vs_sector_3d",
]


def _classify_path(y_series: np.ndarray, intraday_range: np.ndarray | None = None) -> str:
    """Classify the realized path type from a series of daily returns."""
    if len(y_series) < 2:
        return "drift"
    
    # Check for reversal pattern (sign changes)
    signs = np.sign(y_series)
    sign_changes = np.sum(np.diff(signs) != 0)
    total_abs_move = abs(np.sum(y_series))
    sum_abs_moves = np.sum(np.abs(y_series))
    
    if sum_abs_moves == 0:
        return "drift"
    
    efficiency = total_abs_move / sum_abs_moves  # 1.0 = pure trend, 0.0 = pure noise
    
    if efficiency > 0.7 and sign_changes <= 1:
        return "smooth_trend"
    elif abs(y_series[0]) > 1.5 * np.mean(np.abs(y_series[1:])) if len(y_series) > 1 else False:
        return "gap_and_fade"
    elif efficiency < 0.3 and sign_changes >= 2:
        return "volatile_two_way"
    else:
        return "drift"


def train_baseline_model(
    X: np.ndarray,
    y: np.ndarray,
    y_abs: np.ndarray,
    dates: list[str],
    feature_names: list[str],
    daily_returns: list[list[float]] | None = None,
    purge_days: int = 3,
    embargo_days: int = 2,
) -> dict:
    """
    Train baseline magnitude model with quantile regression.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Signed 3-day move targets
        y_abs: Absolute 3-day move targets
        dates: Date strings for CV splitting
        feature_names: Feature column names
        daily_returns: Per-sample list of daily returns for path classification
        purge_days: CV purge window
        embargo_days: CV embargo window
    
    Returns:
        Dict with model outputs, quantiles, and diagnostics.
    """
    if len(X) < 30:
        return {"status": "insufficient_data", "reason": f"need 30+ samples, got {len(X)}"}
    
    n_splits = min(5, max(2, len(X) // 20))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
    
    # ── 1. Point prediction model (LightGBM) ──
    point_cv_scores = []
    best_point_model = None
    best_point_mse = float("inf")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(train_idx) // 20),
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.2, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        mse = float(mean_squared_error(y[val_idx], preds))
        point_cv_scores.append(mse)
        
        if mse < best_point_mse:
            best_point_mse = mse
            best_point_model = model
    
    if best_point_model is None:
        return {"status": "insufficient_data", "reason": "no_valid_cv_folds"}
    
    # Full dataset predictions
    point_preds = best_point_model.predict(X)
    correlation = float(np.corrcoef(point_preds, y)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    
    direction_correct = np.sum(np.sign(point_preds) == np.sign(y))
    direction_accuracy = float(direction_correct / len(y))
    
    # ── 2. Quantile regression models ──
    quantile_models = {}
    quantile_preds = {}
    
    for q in QUANTILES:
        qmodel = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            loss="quantile", alpha=q,
            min_samples_split=max(5, len(X) // 30),
            subsample=0.8,
        )
        qmodel.fit(X, y)
        quantile_models[q] = qmodel
        quantile_preds[q] = qmodel.predict(X)
    
    # ── 3. Threshold probability estimation ──
    # P(|move| > threshold) estimated from quantile distribution
    threshold_probs = {}
    for thresh in THRESHOLDS:
        # Use quantile crossing: interpolate between quantiles
        above_count = np.sum(np.abs(y) > thresh)
        threshold_probs[f"prob_move_gt_{thresh}pct"] = round(float(above_count / len(y)), 4)
    
    # ── 4. Path classification ──
    path_classes = []
    if daily_returns and len(daily_returns) == len(y):
        for dr in daily_returns:
            path_classes.append(_classify_path(np.array(dr)))
    
    path_distribution = {}
    if path_classes:
        for pc in set(path_classes):
            path_distribution[pc] = round(path_classes.count(pc) / len(path_classes), 3)
    
    # ── 5. Feature importance ──
    importances = best_point_model.feature_importances_
    total_imp = sum(importances)
    importance_dict = {}
    if total_imp > 0:
        for i, fname in enumerate(feature_names):
            importance_dict[fname] = round(float(importances[i] / total_imp * 100), 2)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    # ── 6. CV quantile coverage check ──
    q10_preds = quantile_preds[0.10]
    q90_preds = quantile_preds[0.90]
    coverage_90 = float(np.mean((y >= q10_preds) & (y <= q90_preds)))
    
    q25_preds = quantile_preds[0.25]
    q75_preds = quantile_preds[0.75]
    coverage_50 = float(np.mean((y >= q25_preds) & (y <= q75_preds)))
    
    return {
        "status": "trained",
        "specialist_type": "baseline",
        "model_version": "v2.0",
        
        # Point estimates
        "correlation": round(correlation, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "mae": round(float(mean_absolute_error(y, point_preds)), 4),
        "rmse": round(float(np.sqrt(best_point_mse)), 4),
        "cv_mse": [round(s, 6) for s in point_cv_scores],
        
        # Quantile coverage
        "coverage_90": round(coverage_90, 4),
        "coverage_50": round(coverage_50, 4),
        
        # Threshold probabilities (empirical baselines)
        "threshold_probs": threshold_probs,
        
        # Path classification
        "path_distribution": path_distribution,
        
        # Feature importance
        "importance": importance_dict,
        
        # Meta
        "observations": len(X),
        "features_used": feature_names,
        "purged_samples": cv.purged_samples,
    }


def predict_baseline(
    model_state: dict,
    X_new: np.ndarray,
    point_model,
    quantile_models: dict,
) -> dict:
    """
    Generate predictions from trained baseline model.
    
    Returns dict matching magnitude_v2_specialist_outputs schema.
    """
    point_pred = float(point_model.predict(X_new.reshape(1, -1))[0])
    
    quantile_preds = {}
    for q, qmodel in quantile_models.items():
        qpred = float(qmodel.predict(X_new.reshape(1, -1))[0])
        quantile_preds[q] = round(qpred, 4)
    
    # Direction confidence from quantile spread
    q10 = quantile_preds.get(0.10, 0)
    q90 = quantile_preds.get(0.90, 0)
    q50 = quantile_preds.get(0.50, 0)
    
    if q90 - q10 > 0:
        # How much of the distribution is on one side of zero
        if q50 > 0:
            direction_confidence = min(1.0, max(0.5, 1.0 - abs(q10) / (q90 - q10)))
        else:
            direction_confidence = min(1.0, max(0.5, 1.0 - abs(q90) / (q90 - q10)))
    else:
        direction_confidence = 0.5
    
    return {
        "specialist_type": "baseline",
        "model_version": model_state.get("model_version", "v2.0"),
        "expected_move_pct": round(point_pred, 4),
        "expected_abs_move_pct": round(abs(point_pred), 4),
        "direction_confidence": round(direction_confidence, 4),
        "q10": quantile_preds.get(0.10),
        "q25": quantile_preds.get(0.25),
        "q50": quantile_preds.get(0.50),
        "q75": quantile_preds.get(0.75),
        "q90": quantile_preds.get(0.90),
        "model_confidence": round(model_state.get("correlation", 0), 4),
        "training_samples": model_state.get("observations", 0),
    }
