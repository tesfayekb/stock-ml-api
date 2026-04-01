"""
Event-Driven Magnitude Model — predicts catalyst-driven move magnitude.

Handles non-earnings catalysts: analyst upgrades/downgrades, FDA decisions,
macro events, guidance revisions, institutional flow spikes, M&A, etc.

Key difference from baseline: event features dominate, and the model
learns event-type-specific magnitude patterns.
"""

import numpy as np
import logging
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cv import PurgedWalkForwardCV

log = logging.getLogger("magnitude-v2")

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

EVENT_FEATURES = [
    # Market context
    "vix_level", "vix_term_slope", "credit_spread",
    "spy_return_1d",
    "realized_vol_5d", "realized_vol_10d",
    
    # Stock context
    "rsi14", "momentum_3d", "atr_pct",
    "volume_ratio_20d", "market_beta",
    
    # Event-specific
    "event_type_encoded",       # Categorical encoding of event type
    "event_magnitude",          # Raw event magnitude (from events table)
    "event_direction_encoded",  # +1 bullish, -1 bearish, 0 neutral
    "catalyst_quality",         # 0-100 quality score
    "catalyst_freshness_encoded",  # new=4, confirmed=3, rumored=2, recycled=1
    "event_crowding_score",     # How many competing events
    "catalyst_count_7d",        # Recent catalyst density
    "historical_event_response", # Historical avg move for this event type + ticker
    "sector_relative_strength",
    "pre_event_drift_5d",       # Drift before event (positioning proxy)
    "short_interest_ratio",
]

# Event type encoding
EVENT_TYPE_MAP = {
    "earnings_report": 1, "guidance_change": 2, "analyst_upgrade": 3,
    "analyst_downgrade": 4, "fda_decision": 5, "macro_data": 6,
    "m_and_a": 7, "buyback": 8, "insider_trade": 9,
    "institutional_flow": 10, "dividend_change": 11, "sector_rotation": 12,
    "short_squeeze": 13, "technical_breakout": 14,
}


def encode_event_type(event_type: str) -> int:
    """Encode event type string to integer for model input."""
    return EVENT_TYPE_MAP.get(event_type, 0)


def train_event_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    feature_names: list[str],
    event_types: list[str] | None = None,
    purge_days: int = 3,
    embargo_days: int = 2,
) -> dict:
    """
    Train event-driven magnitude model.
    
    Similar structure to baseline but with event-specific features
    and per-event-type diagnostics.
    """
    if len(X) < 30:
        return {"status": "insufficient_data", "reason": f"need 30+ event samples, got {len(X)}"}
    
    n_splits = min(5, max(2, len(X) // 15))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
    
    # ── Point model ──
    cv_scores = []
    best_model = None
    best_mse = float("inf")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(train_idx) // 15),
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.15, reg_lambda=0.2, verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
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
            n_estimators=100, learning_rate=0.05, max_depth=3,
            loss="quantile", alpha=q,
            min_samples_split=max(5, len(X) // 20),
        )
        qmodel.fit(X, y)
        quantile_preds[q] = qmodel.predict(X)
    
    coverage_90 = float(np.mean(
        (y >= quantile_preds[0.10]) & (y <= quantile_preds[0.90])
    ))
    
    # ── Per-event-type diagnostics ──
    event_type_metrics = {}
    if event_types and len(event_types) == len(y):
        for et in set(event_types):
            mask = [i for i, e in enumerate(event_types) if e == et]
            if len(mask) >= 5:
                et_preds = point_preds[mask]
                et_actuals = y[mask]
                event_type_metrics[et] = {
                    "n": len(mask),
                    "mae": round(float(mean_absolute_error(et_actuals, et_preds)), 4),
                    "avg_abs_move": round(float(np.mean(np.abs(et_actuals))), 2),
                    "correlation": round(float(np.corrcoef(et_preds, et_actuals)[0, 1]) 
                                        if len(mask) > 2 else 0, 4),
                }
    
    importances = best_model.feature_importances_
    total_imp = sum(importances)
    importance_dict = {}
    if total_imp > 0:
        for i, fname in enumerate(feature_names):
            importance_dict[fname] = round(float(importances[i] / total_imp * 100), 2)
    
    return {
        "status": "trained",
        "specialist_type": "event",
        "model_version": "v2.0",
        "correlation": round(correlation, 4),
        "mae": round(float(mean_absolute_error(y, point_preds)), 4),
        "coverage_90": round(coverage_90, 4),
        "cv_mse": [round(s, 6) for s in cv_scores],
        "event_type_metrics": event_type_metrics,
        "importance": importance_dict,
        "observations": len(X),
        "features_used": feature_names,
    }
