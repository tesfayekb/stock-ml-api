"""
Earnings-Specific Magnitude Model — predicts post-earnings moves using
features unique to earnings events: surprise history, implied move,
guidance revisions, sector earnings density.

Earnings moves behave fundamentally differently from non-earnings moves:
- Higher kurtosis (fat tails)
- Gap-dominated (most move happens overnight)
- Mean-reversion after initial gap is weaker
- Implied move from options straddle provides strong prior
"""

import numpy as np
import logging
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from cv import PurgedWalkForwardCV

log = logging.getLogger("magnitude-v2")

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Earnings-specific features
EARNINGS_FEATURES = [
    # Market context
    "vix_level", "vix_term_slope",
    "realized_vol_10d", "realized_vol_20d",
    "iv_rv_spread",
    
    # Stock context
    "rsi14", "momentum_5d", "momentum_10d",
    "atr_pct", "volume_ratio_20d",
    "market_beta",
    
    # Earnings-specific
    "options_implied_move",        # Straddle-implied expected move
    "pre_earnings_drift_5d",       # Stock drift in 5 days before earnings
    "historical_surprise_avg",     # Average EPS surprise magnitude
    "historical_post_earn_move",   # Average post-earnings absolute move
    "guidance_revision_pct",       # Recent analyst revision direction
    "earnings_density_sector",     # How many sector peers report same week
    "short_interest_ratio",        # SI as % of float
    "days_since_last_earnings",    # Days since prior quarter
    "beat_streak",                 # Consecutive quarters beating estimates
    "sector_earnings_momentum",    # Sector earnings trend
]


def _extract_earnings_features(
    base_features: dict,
    earnings_history: dict,
) -> dict:
    """Combine base features with earnings-specific enrichments."""
    ef = {}
    
    # Pull base features
    for feat in EARNINGS_FEATURES:
        if feat in base_features:
            ef[feat] = base_features[feat]
    
    # Earnings history enrichments
    if earnings_history:
        surprises = earnings_history.get("surprise_history", [])
        if surprises:
            ef["historical_surprise_avg"] = float(np.mean([abs(s) for s in surprises[-8:]]))
        
        moves = earnings_history.get("post_earnings_moves", [])
        if moves:
            ef["historical_post_earn_move"] = float(np.mean([abs(m) for m in moves[-8:]]))
        
        ef["beat_streak"] = earnings_history.get("beat_streak", 0)
        ef["days_since_last_earnings"] = earnings_history.get("days_since_last", 90)
        ef["guidance_revision_pct"] = earnings_history.get("revision_pct", 0)
        ef["sector_earnings_momentum"] = earnings_history.get("sector_momentum", 0)
    
    # Derive pre-earnings drift from momentum
    ef["pre_earnings_drift_5d"] = base_features.get("momentum_5d", 0)
    
    return ef


def train_earnings_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    feature_names: list[str],
    purge_days: int = 5,
    embargo_days: int = 3,
    raw_data: list | None = None,
) -> dict:
    """
    Train earnings-specific magnitude model.
    
    Uses wider purge/embargo because earnings moves have longer settling.
    Requires fewer samples (50+) since earnings are rarer events.
    """
    if len(X) < 20:
        return {"status": "insufficient_data", "reason": f"need 20+ earnings samples, got {len(X)}"}
    
    n_splits = min(4, max(2, len(X) // 12))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
    
    # ── Point model ──
    cv_scores = []
    best_model = None
    best_mse = float("inf")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        if len(train_idx) < 10:
            continue
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            num_leaves=10, min_child_samples=max(2, len(train_idx) // 10),
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.3, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        from sklearn.metrics import mean_squared_error
        mse = float(mean_squared_error(y[val_idx], preds))
        cv_scores.append(mse)
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
    
    if best_model is None:
        return {"status": "insufficient_data", "reason": "no_valid_cv_folds"}
    
    point_preds = best_model.predict(X)
    correlation = float(np.corrcoef(point_preds, y)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    
    # ── Quantile models ──
    quantile_preds = {}
    for q in QUANTILES:
        qmodel = GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=3,
            loss="quantile", alpha=q,
            min_samples_split=max(3, len(X) // 15),
        )
        qmodel.fit(X, y)
        quantile_preds[q] = qmodel.predict(X)
    
    # Coverage
    q10 = quantile_preds[0.10]
    q90 = quantile_preds[0.90]
    coverage_90 = float(np.mean((y >= q10) & (y <= q90)))
    
    # Earnings-specific diagnostics
    abs_y = np.abs(y)
    kurtosis = float(np.mean((y - np.mean(y))**4) / (np.std(y)**4)) if np.std(y) > 0 else 3.0
    
    importances = best_model.feature_importances_
    total_imp = sum(importances)
    importance_dict = {}
    if total_imp > 0:
        for i, fname in enumerate(feature_names):
            importance_dict[fname] = round(float(importances[i] / total_imp * 100), 2)
    
    return {
        "status": "trained",
        "specialist_type": "earnings",
        "model_version": "v2.0",
        "correlation": round(correlation, 4),
        "mae": round(float(mean_absolute_error(y, point_preds)), 4),
        "coverage_90": round(coverage_90, 4),
        "cv_mse": [round(s, 6) for s in cv_scores],
        "kurtosis": round(kurtosis, 2),
        "avg_abs_move": round(float(np.mean(abs_y)), 2),
        "importance": importance_dict,
        "observations": len(X),
        "features_used": feature_names,
    }
    
    def predict_earnings(model_state: dict, X: np.ndarray) -> "SpecialistOutput":
    """Predict using stored earnings specialist model."""
    from .meta_model import SpecialistOutput
    
    point_model = model_state.get("point_model")
    quantile_models = model_state.get("quantile_models", {})
    
    if point_model is None:
        return SpecialistOutput(
            specialist_type="earnings",
            point_estimate=0.0,
            quantiles={},
            confidence=0.0,
            weight_hint=0.0,
        )
    
    X_2d = X.reshape(1, -1) if X.ndim == 1 else X
    point_pred = float(point_model.predict(X_2d)[0])
    
    quantiles = {}
    for q, qmodel in quantile_models.items():
        quantiles[q] = float(qmodel.predict(X_2d)[0])
    
    ci_width = quantiles.get(0.90, point_pred + 1) - quantiles.get(0.10, point_pred - 1)
    confidence = max(0.0, min(1.0, 1.0 - (ci_width / (abs(point_pred) + 1e-6)) * 0.1))
    
    return SpecialistOutput(
        specialist_type="earnings",
        point_estimate=point_pred,
        quantiles=quantiles,
        confidence=round(confidence, 4),
        weight_hint=1.2,  # Earnings specialist gets slight boost near earnings
    )
