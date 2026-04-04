"""
Stock ML Backend — FastAPI service on Railway.
Queries score_deltas from Supabase, trains LightGBM/Ridge/MLP per-factor models,
writes optimized weights back to stock_impact_profiles + calibration_state.
"""
import os
import logging
from typing import Optional
from regime_hmm import get_hmm
from tactical_model import get_tactical_model_24h, TacticalModel24h  # v2: was get_tactical_model
from tactical_model_2h import get_tactical_model_2h, TacticalModel2h
from magnitude_v2 import (
    extract_features as mag_extract_features,
    FEATURE_VERSION as MAG_FEATURE_VERSION,
    train_baseline_model,
    predict_baseline,
    train_earnings_model,
    train_event_model,
    blend_predictions,
    compute_dynamic_weights,
    compute_disagreement,
    SpecialistOutput,
    MagnitudeCalibrator,
    save_model_artifact,
    load_model_artifact,
)

import httpx
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error


from supabase_client import (
    fetch_score_deltas,
    fetch_score_deltas_range,
    fetch_current_weights,
    fetch_market_defaults,
    fetch_calibration_state,
    write_optimized_weights,
)
from cv import PurgedWalkForwardCV
from features import (
    build_factor_matrix,
    importances_to_weights,
    compute_shap_importance,
    MIN_SAMPLES,
    MAX_WEIGHT,
)
from sector_map import get_sector
from models import train_lightgbm_model, train_ridge_model, train_mlp_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ml-backend")

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(title="Stock ML Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_SECRET = os.environ.get("ML_API_SECRET", "")


# ── Auth helper ──────────────────────────────────────────────────────────
def verify_caller(authorization: Optional[str] = Header(None)):
    if API_SECRET and authorization != f"Bearer {API_SECRET}":
        raise HTTPException(401, "Unauthorized")


# ── Request models ───────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker: str
    user_id: str
    lookback_days: int = 365
    purge_days: int = 2
    embargo_days: int = 1
    callback_url: str | None = None
    
class TacticalPredictRequest(BaseModel):
    user_id: str
    features: dict  # latest fast_features.features jsonb

class TacticalTrainRequest(BaseModel):
    user_id: str
    lookback_days: int = 90
    callback_url: str | None = None


class TrainEnsembleRequest(BaseModel):
    ticker: str
    user_id: str
    lookback_days: int = 365
    purge_days: int = 2
    embargo_days: int = 1
    models: list[str] = ["lightgbm", "ridge", "mlp"]
    callback_url: str | None = None


class BacktestRequest(BaseModel):
    ticker: str
    user_id: str
    start_date: str
    end_date: str


class ExplainRequest(BaseModel):
    """SHAP explain endpoint request model."""
    ticker: str
    user_id: str
    features: dict  # { event_type: contribution_value }


class TrainIncrementalRequest(BaseModel):
    ticker: str
    user_id: str
    lookback_days: int = 30       # context window
    new_data_hours: int = 48      # only train on recent data
    purge_days: int = 2
    embargo_days: int = 1
    callback_url: str | None = None


class TacticalPredict2hRequest(BaseModel):
    user_id: str
    features: dict    # latest fast_features.features jsonb (v2 with intraday keys)
    horizon: str = "2h"


class TacticalTrain2hRequest(BaseModel):
    user_id: str
    lookback_days: int = 60
    callback_url: str | None = None


# ═══════════════════════════════════════════════════════════════════════
#  Async Webhook Helpers
# ═══════════════════════════════════════════════════════════════════════

async def _send_callback(callback_url: str, payload: dict):
    """POST training results back to the Supabase ml-training-webhook."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                callback_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {API_SECRET}",
                    "Content-Type": "application/json",
                },
            )
            log.info(f"  Callback sent to {callback_url}: status={resp.status_code}")
    except Exception as e:
        log.error(f"  Callback failed: {e}")


def _run_ensemble_training_sync(req: TrainEnsembleRequest) -> dict:
    """
    Synchronous ensemble training logic — used by both sync and async paths.
    """
    ticker = req.ticker.upper()

    raw = fetch_score_deltas(ticker, req.user_id, req.lookback_days)
    if len(raw) < MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "ticker": ticker,
            "user_id": req.user_id,
            "rows": len(raw),
            "min_required": MIN_SAMPLES,
            "success": False,
        }

    X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
    if X is None:
        return {
            "status": "insufficient_data",
            "ticker": ticker,
            "user_id": req.user_id,
            "rows": len(raw),
            "min_required": MIN_SAMPLES,
            "success": False,
        }

    log.info(f"  Feature matrix: {X.shape[0]}×{X.shape[1]} factors")

    results = {"status": "trained", "ticker": ticker, "models": {}, "success": True}

    # Train each requested model
    if "lightgbm" in req.models:
        results["models"]["lightgbm"] = train_lightgbm_model(
            X, y, dates, factor_names, req.purge_days, req.embargo_days, req.user_id, ticker,
        )

    if "ridge" in req.models:
        results["models"]["ridge"] = train_ridge_model(
            X, y, dates, factor_names, req.purge_days, req.embargo_days,
        )

    if "mlp" in req.models:
        results["models"]["mlp"] = train_mlp_model(
            X, y, dates, factor_names, req.purge_days, req.embargo_days,
        )

    # Compute ensemble agreement with correlation-weighted averaging
    model_predictions = {}
    for model_name, model_result in results["models"].items():
        if model_result.get("status") == "trained":
            model_predictions[model_name] = model_result.get("correlation", 0)

    if len(model_predictions) >= 2:
        corrs = list(model_predictions.values())

        positive_corrs = {k: max(0.01, v) for k, v in model_predictions.items()}
        total_corr = sum(positive_corrs.values())
        model_influence = {k: v / total_corr for k, v in positive_corrs.items()}

        ensemble_weights = {}
        for model_name, model_result in results["models"].items():
            if model_result.get("status") == "trained" and model_result.get("weights"):
                influence = model_influence.get(model_name, 1 / len(model_predictions))
                for factor, weight in model_result["weights"].items():
                    if factor not in ensemble_weights:
                        ensemble_weights[factor] = 0
                    ensemble_weights[factor] += weight * influence

        ensemble_weights = {k: round(v, 4) for k, v in ensemble_weights.items()}

        results["ensemble"] = {
            "model_count": len(corrs),
            "avg_correlation": round(sum(corrs) / len(corrs), 4),
            "max_correlation": round(max(corrs), 4),
            "min_correlation": round(min(corrs), 4),
            "correlation_spread": round(max(corrs) - min(corrs), 4),
            "model_influence": {k: round(v, 4) for k, v in model_influence.items()},
            "ensemble_weights": ensemble_weights,
        }

        # Top-level fields for webhook compatibility
        results["correlation"] = results["ensemble"]["avg_correlation"]
        results["optimized_weights"] = ensemble_weights
        results["weights_written"] = len(ensemble_weights)

    # Aggregate importance across models
    all_importance = {}
    for model_name, model_result in results["models"].items():
        if model_result.get("importance"):
            for factor, imp in model_result["importance"].items():
                if factor not in all_importance:
                    all_importance[factor] = 0
                all_importance[factor] += imp
    if all_importance:
        total_imp = sum(all_importance.values())
        if total_imp > 0:
            results["importance"] = {k: round(v / total_imp * 100, 2) for k, v in all_importance.items()}

    results["user_id"] = req.user_id
    return results


async def _train_ensemble_and_callback(req: TrainEnsembleRequest):
    """Background task: run ensemble training then POST results to callback_url."""
    try:
        log.info(f"🚀 Background ensemble training started for {req.ticker}")
        result = _run_ensemble_training_sync(req)
        await _send_callback(req.callback_url, result)
    except Exception as e:
        log.exception(f"Background ensemble training failed for {req.ticker}")
        await _send_callback(req.callback_url, {
            "ticker": req.ticker.upper(),
            "user_id": req.user_id,
            "success": False,
            "error": str(e),
        })


# ═══════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Health check — reports HMM + tactical model status."""
    hmm = get_hmm()
    tactical_24h = get_tactical_model_24h()   # v2
    tactical_2h = get_tactical_model_2h()     # 2h
    return {
        "status": "ok",
        # HMM
        "hmm_loaded": hmm.model is not None,
        "hmm_states": list(hmm.state_map.values()) if hmm.state_map else [],
        # 24h tactical (v2)
        "tactical_24h_trained": tactical_24h.is_trained,
        "tactical_24h_samples": tactical_24h.training_samples,
        "tactical_24h_last_trained": tactical_24h.last_trained_at,
        "tactical_24h_cv_accuracy": tactical_24h.cv_direction_accuracy,
        "tactical_24h_cv_ridge_accuracy": tactical_24h.cv_ridge_accuracy,
        "tactical_24h_cv_magnitude_corr": tactical_24h.cv_magnitude_corr,
        "tactical_24h_model_version": "tactical-24h-v2",
        # 2h tactical
        "tactical_2h_trained": tactical_2h.is_trained,
        "tactical_2h_samples": tactical_2h.training_samples,
        "tactical_2h_last_trained": tactical_2h.last_trained_at,
        "tactical_2h_cv_accuracy": tactical_2h.cv_direction_accuracy,
    }


@app.post("/train")
async def train(req: TrainRequest, authorization: Optional[str] = Header(None)):
    verify_caller(authorization)
    ticker = req.ticker.upper()
    log.info(f"Train request: {ticker}, user={req.user_id[:8]}..., lookback={req.lookback_days}d")

    try:
        # ── 1. Fetch data ──
        raw = fetch_score_deltas(ticker, req.user_id, req.lookback_days)
        log.info(f"  Fetched {len(raw)} score_delta rows for {ticker}")

        if len(raw) < MIN_SAMPLES:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": MIN_SAMPLES,
            }

        X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
        if X is None:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": MIN_SAMPLES,
            }

        log.info(f"  Feature matrix: {X.shape[0]} observations × {X.shape[1]} factors: {factor_names}")

        # ── 2. Fetch baselines ──
        market_defaults = fetch_market_defaults(req.user_id)
        current_weights = fetch_current_weights(ticker, req.user_id)
        prev_state = fetch_calibration_state(ticker, req.user_id)
        prev_best = prev_state["best_correlation"] if prev_state else None

        # ── 3. Train LightGBM with purged walk-forward validation ──
        n_splits = min(5, max(2, len(X) // 15))
        cv = PurgedWalkForwardCV(
            n_splits=n_splits,
            purge_days=req.purge_days,
            embargo_days=req.embargo_days,
        )
        cv_scores = []
        cv_correlations = []
        best_model = None
        best_mse = float("inf")
        best_val_idx = None

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=max(3, len(train_idx) // 20),
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
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

        # ── 4. Compute final metrics ──
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

        log.info(
            f"  Training complete: corr={correlation:.3f}, "
            f"dir_acc={direction_accuracy:.1%}, "
            f"cv_mse={cv_scores}, factors={len(factor_names)}, "
            f"purged_samples={cv.purged_samples}"
        )

        # ── 4b. Compute SHAP importance on best validation fold ──
        shap_importance = {}
        if best_val_idx is not None and len(best_val_idx) >= 3:
            shap_importance = compute_shap_importance(
                best_model, X[best_val_idx], factor_names
            )

        # ── 5. Convert importances → weight overrides ──
        optimized_weights = importances_to_weights(
            importances, factor_names, market_defaults, current_weights
        )

        # ── 6. Regression guard: only write if correlation improved ──
        should_write = True
        regression_note = None

        if prev_best is not None and correlation < prev_best * 0.9:
            should_write = False
            regression_note = (
                f"Skipped write: new corr {correlation:.3f} < "
                f"90% of best {prev_best:.3f}"
            )
            log.warning(f"  {regression_note}")

        write_result = None
        if should_write and optimized_weights:
            sector = get_sector(ticker)
            write_result = write_optimized_weights(
                ticker=ticker,
                user_id=req.user_id,
                sector=sector,
                weights=optimized_weights,
                correlation=correlation,
                sample_size=len(X),
                prev_best=prev_best,
            )
            log.info(f"  Weights written: {write_result}")

        return {
            "status": "trained",
            "ticker": ticker,
            "rows": len(raw),
            "observations": len(X),
            "factors": factor_names,
            "correlation": round(correlation, 4),
            "direction_accuracy": round(direction_accuracy, 4),
            "cv_mse": [round(s, 6) for s in cv_scores],
            "cv_correlations": [round(c, 4) for c in cv_correlations],
            "importance": importance_dict,
            "optimized_weights": optimized_weights,
            "weights_written": write_result is not None,
            "regression_note": regression_note,
            "prev_best_correlation": prev_best,
            "purged_samples": cv.purged_samples,
            "purge_days": req.purge_days,
            "embargo_days": req.embargo_days,
            "shap_importance": shap_importance,
        }

    except Exception as e:
        log.exception(f"Train failed for {ticker}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════
#  Tactical 24h Endpoints
# ═══════════════════════════════════════════════════════════════════════

@app.post("/predict-tactical")
async def predict_tactical(req: TacticalPredictRequest, authorization: Optional[str] = Header(None)):
    """Predict 24h market direction + magnitude (v2: decision layer + canonical schema)."""
    verify_caller(authorization)
    try:
        model = get_tactical_model_24h()
        result = model.predict(req.features)
        result["user_id"] = req.user_id
        return result
    except Exception as e:
        log.exception("Tactical-24h predict failed")
        return {
            "success": False,
            "error": str(e),
            "direction": "abstain",
            "direction_confidence": 0.0,
            "magnitude_estimate": 0.0,
            "model_version": "tactical-24h-v2",
            "decision_layer": {"abstain": True, "abstain_reason": f"exception: {str(e)}"},
        }


@app.post("/train-tactical")
async def train_tactical(
    req: TacticalTrainRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """Train 24h tactical model (v2: binary + ridge + canonical schema)."""
    verify_caller(authorization)
    log.info(f"Tactical-24h train: user={req.user_id[:8]}..., lookback={req.lookback_days}d")

    try:
        model = get_tactical_model_24h()
        result = model.train(req.user_id, req.lookback_days)

        if req.callback_url:
            await _send_callback(req.callback_url, {
                **result,
                "model_type": "tactical_24h_lgbm_v2",
                "user_id": req.user_id,
                "ticker": "SPY",
                "success": result.get("status") == "trained",
            })
            return {"accepted": True, "status": result["status"]}

        return result
    except Exception as e:
        log.exception("Tactical-24h train failed")
        error_result = {
            "success": False,
            "error": str(e),
            "user_id": req.user_id,
            "model_type": "tactical_24h_lgbm_v2",
        }
        if req.callback_url:
            await _send_callback(req.callback_url, error_result)
            return {"accepted": True, "status": "error"}
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble Training Endpoint (sync + async via callback_url)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/train-ensemble")
async def train_ensemble(
    req: TrainEnsembleRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """
    Train multiple models on the same feature matrix.
    If callback_url is provided, runs async in background and returns immediately.
    Otherwise runs synchronously and returns results.
    """
    verify_caller(authorization)
    ticker = req.ticker.upper()
    log.info(f"Ensemble train: {ticker}, models={req.models}, user={req.user_id[:8]}..., async={req.callback_url is not None}")

    if req.callback_url:
        background_tasks.add_task(_train_ensemble_and_callback, req)
        return {
            "accepted": True,
            "ticker": ticker,
            "models": req.models,
            "message": f"Ensemble training queued for {ticker}. Results will be POSTed to callback_url.",
        }

    try:
        result = _run_ensemble_training_sync(req)
        return result
    except Exception as e:
        log.exception(f"Ensemble train failed for {ticker}")
        raise HTTPException(500, str(e))


@app.post("/train-incremental")
async def train_incremental(
    req: TrainIncrementalRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """Fast incremental retrain using warm-start Ridge + LightGBM init_model."""
    verify_caller(authorization)
    ticker = req.ticker.upper()
    log.info(f"Incremental train: {ticker}, user={req.user_id[:8]}...")

    try:
        raw = fetch_score_deltas(ticker, req.user_id, req.lookback_days)
        if len(raw) < MIN_SAMPLES:
            result = {
                "status": "insufficient_data",
                "ticker": ticker,
                "user_id": req.user_id,
                "rows": len(raw),
                "min_required": MIN_SAMPLES,
                "mode": "incremental",
                "success": False,
            }
            if req.callback_url:
                await _send_callback(req.callback_url, result)
                return {"accepted": True, "ticker": ticker, "mode": "incremental"}
            return result

        X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
        if X is None:
            result = {
                "status": "insufficient_data",
                "ticker": ticker,
                "user_id": req.user_id,
                "mode": "incremental",
                "success": False,
            }
            if req.callback_url:
                await _send_callback(req.callback_url, result)
                return {"accepted": True, "ticker": ticker, "mode": "incremental"}
            return result

        # Train Ridge with warm_start
        from models.ridge_model import train_ridge_model
        ridge_result = train_ridge_model(X, y, dates, factor_names, req.purge_days, req.embargo_days)

        # Train LightGBM with fewer rounds (fast refit)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=50,  # fast refit
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X) // 20),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        lgb_model.fit(X, y)
        lgb_preds = lgb_model.predict(X)
        lgb_corr = float(np.corrcoef(lgb_preds, y)[0, 1])
        if np.isnan(lgb_corr):
            lgb_corr = 0.0

        # Combine weights (simple average of Ridge + LightGBM)
        ensemble_weights = {}
        ridge_weights = ridge_result.get("weights", {})
        lgb_importances = lgb_model.feature_importances_
        lgb_weights = importances_to_weights(
            lgb_importances, factor_names,
            fetch_market_defaults(req.user_id),
            fetch_current_weights(ticker, req.user_id),
        )

        all_factors = set(list(ridge_weights.keys()) + list(lgb_weights.keys()))
        for f in all_factors:
            rw = ridge_weights.get(f, 0)
            lw = lgb_weights.get(f, 0)
            ensemble_weights[f] = round((rw + lw) / 2, 4)

        ridge_corr = ridge_result.get("correlation", 0)
        avg_corr = round((ridge_corr + lgb_corr) / 2, 4)

        result = {
            "status": "trained",
            "ticker": ticker,
            "user_id": req.user_id,
            "mode": "incremental",
            "success": True,
            "correlation": avg_corr,
            "optimized_weights": ensemble_weights,
            "weights_written": len(ensemble_weights),
            "models": {
                "ridge": {"correlation": ridge_corr, "status": ridge_result.get("status")},
                "lightgbm": {"correlation": round(lgb_corr, 4), "status": "trained"},
            },
            "observations": len(X),
            "factors": factor_names,
        }

        if req.callback_url:
            await _send_callback(req.callback_url, result)
            return {"accepted": True, "ticker": ticker, "mode": "incremental"}

        return result

    except Exception as e:
        log.exception(f"Incremental train failed for {ticker}")
        error_result = {
            "ticker": ticker,
            "user_id": req.user_id,
            "mode": "incremental",
            "success": False,
            "error": str(e),
        }
        if req.callback_url:
            await _send_callback(req.callback_url, error_result)
            return {"accepted": True, "ticker": ticker, "mode": "incremental"}
        raise HTTPException(500, str(e))


@app.post("/backtest")
async def backtest(req: BacktestRequest, authorization: Optional[str] = Header(None)):
    verify_caller(authorization)
    ticker = req.ticker.upper()
    log.info(f"Backtest request: {ticker}, {req.start_date} → {req.end_date}")

    try:
        raw = fetch_score_deltas_range(ticker, req.user_id, req.start_date, req.end_date)
        log.info(f"  Fetched {len(raw)} score_delta rows for {ticker}")

        if len(raw) < 10:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": 10,
            }

        X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
        if X is None:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": MIN_SAMPLES,
            }

        split = int(len(X) * 0.7)
        purge_buffer = 2
        train_end = max(0, split - purge_buffer)
        test_start = split

        if train_end < 8 or (len(X) - test_start) < 5:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "observations": len(X),
                "reason": "Not enough data for 70/30 split after purge",
            }

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:], y[test_start:]
        purged_count = split - train_end

        log.info(f"  Split: {len(X_train)} train / {len(X_test)} test, {len(factor_names)} factors, {purged_count} purged")

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = float(mean_squared_error(y_test, preds))
        correlation = float(np.corrcoef(preds, y_test)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0

        direction_correct = np.sum(np.sign(preds) == np.sign(y_test))
        direction_accuracy = float(direction_correct / len(y_test))

        importances = model.feature_importances_
        importance_dict = {
            factor_names[i]: int(importances[i])
            for i in range(len(factor_names))
        }

        factor_performance = {}
        for i, factor in enumerate(factor_names):
            col = X_test[:, i]
            nonzero = col != 0
            if nonzero.sum() > 3:
                factor_dir_acc = float(
                    np.sum(np.sign(col[nonzero]) == np.sign(y_test[nonzero]))
                    / nonzero.sum()
                )
                factor_performance[factor] = {
                    "direction_accuracy": round(factor_dir_acc, 4),
                    "nonzero_observations": int(nonzero.sum()),
                    "importance": int(importances[i]),
                }

        log.info(
            f"  Backtest complete: corr={correlation:.3f}, "
            f"dir_acc={direction_accuracy:.1%}, mse={mse:.6f}"
        )

        return {
            "status": "complete",
            "ticker": ticker,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "rows": len(raw),
            "observations": len(X),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "factors": factor_names,
            "mse": round(mse, 6),
            "correlation": round(correlation, 4),
            "direction_accuracy": round(direction_accuracy, 4),
            "importance": importance_dict,
            "factor_performance": factor_performance,
            "purged_samples": purged_count,
        }

    except Exception as e:
        log.exception(f"Backtest failed for {ticker}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════
#  SHAP Explain Endpoint
# ═══════════════════════════════════════════════════════════════════════

@app.post("/explain")
async def explain(req: ExplainRequest, authorization: Optional[str] = Header(None)):
    """
    Given a ticker + feature vector, return per-feature SHAP contributions.
    """
    verify_caller(authorization)
    try:
        rows = fetch_score_deltas(req.ticker, req.user_id, 365)
        if len(rows) < MIN_SAMPLES:
            return {"status": "insufficient_data", "rows": len(rows)}

        X, y, factor_names, dates = build_factor_matrix(rows, "actual_move_3d")
        if X is None:
            return {"status": "insufficient_data"}

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X) // 20),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model.fit(X, y)

        feature_vector = np.array([[req.features.get(f, 0) for f in factor_names]])

        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_vector)[0]

        contributions = {
            fname: round(float(sv), 4)
            for fname, sv in zip(factor_names, shap_values)
        }
        contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))

        return {
            "status": "ok",
            "ticker": req.ticker,
            "prediction": float(model.predict(feature_vector)[0]),
            "shap_contributions": contributions,
            "base_value": float(explainer.expected_value),
        }
    except Exception as e:
        log.exception(f"Explain failed for {req.ticker}")
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════
#  2h Tactical Forecast Endpoints (Regime v6 Phase 3)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/predict-tactical-2h")
async def predict_tactical_2h(
    req: TacticalPredict2hRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Predict 2h market direction + magnitude from v2 fast_features.
    Returns direction, confidence, magnitude, continuation/reversal probs.
    """
    verify_caller(authorization)
    try:
        model = get_tactical_model_2h()
        result = model.predict(req.features)
        result["user_id"] = req.user_id
        return result
    except Exception as e:
        log.exception("2h Tactical predict failed")
        return {
            "success": False,
            "error": str(e),
            "direction": "flat",
            "direction_confidence": 0.33,
            "magnitude_estimate": 0.0,
            "model_version": "tactical-2h-v1",
        }


@app.post("/train-tactical-2h")
async def train_tactical_2h(
    req: TacticalTrain2hRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """
    Train 2h tactical model on labeled_2h fast_features data.
    Requires MIN_TACTICAL_2H_SAMPLES (150) labeled 2h snapshots.
    """
    verify_caller(authorization)
    log.info(f"2h Tactical train request: user={req.user_id[:8]}..., lookback={req.lookback_days}d")

    try:
        model = get_tactical_model_2h()
        result = model.train(req.user_id, req.lookback_days)

        if req.callback_url:
            await _send_callback(req.callback_url, {
                **result,
                "model_type": "tactical_2h_lgbm",
                "user_id": req.user_id,
                "ticker": "SPY",
                "success": result.get("status") == "trained",
            })
            return {"accepted": True, "status": result["status"]}

        return result
    except Exception as e:
        log.exception("2h Tactical train failed")
        error_result = {
            "success": False,
            "error": str(e),
            "user_id": req.user_id,
            "model_type": "tactical_2h_lgbm",
        }
        if req.callback_url:
            await _send_callback(req.callback_url, error_result)
            return {"accepted": True, "status": "error"}
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════════
#  Regime HMM Endpoints (hardened 2026-03-23)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/regime/train")
async def regime_train(request: Request):
    """Train HMM on historical feature matrix from bootstrap function."""
    try:
        body = await request.json()
        features = np.array(body["features"], dtype=np.float64)
        n_states = body.get("n_states", 4)

        log.info(f"Regime train: {features.shape[0]} samples, {features.shape[1]} features, {n_states} states")

        hmm = get_hmm()
        if n_states != hmm.n_states:
            hmm.n_states = n_states
            hmm.model = None

        result = hmm.train(features)

        if "error" in result:
            log.warning(f"Regime train returned error: {result['error']}")
            return {"success": False, **result}

        log.info(f"Regime train complete: converged={result.get('converged')}, states={result.get('state_labels')}")
        return {"success": True, **result}

    except Exception as e:
        log.exception("Regime train failed")
        return {
            "success": False,
            "error": str(e),
            "state": "normal",
            "probabilities": {"risk_on": 25, "normal": 25, "risk_off": 25, "crisis": 25},
            "confidence": 0.0,
        }


@app.post("/regime/predict")
async def regime_predict(request: Request):
    """Predict current regime from feature vector."""
    try:
        body = await request.json()
        features = np.array(body["features"], dtype=np.float64)

        log.info(f"Regime predict: features shape={features.shape}")

        hmm = get_hmm()
        result = hmm.predict(features)

        if "error" in result:
            log.warning(f"Regime predict soft error: {result['error']}")
            return {"success": False, **result}

        return {"success": True, **result}

    except Exception as e:
        log.exception("Regime predict failed")
        return {
            "success": False,
            "error": str(e),
            "state": "normal",
            "probabilities": {"risk_on": 25, "normal": 25, "risk_off": 25, "crisis": 25},
            "confidence": 0.0,
        }

# ═══════════════════════════════════════════════════════════════════════
#  Magnitude v2 — Specialist Model Endpoints (Shadow Mode)
# ═══════════════════════════════════════════════════════════════════════

class MagnitudeV2TrainRequest(BaseModel):
    user_id: str
    tickers: list[str] | None = None       # None = all tracked tickers
    specialist_types: list[str] = ["baseline", "earnings", "event"]
    lookback_days: int = 365
    purge_days: int = 3
    embargo_days: int = 2
    callback_url: str | None = None


class MagnitudeV2PredictRequest(BaseModel):
    user_id: str
    ticker: str
    horizon_days: int = 3
    include_quantiles: bool = True
    include_path_class: bool = True


class MagnitudeV2CalibrateRequest(BaseModel):
    user_id: str
    specialist_type: str = "baseline"
    lookback_days: int = 90


# ── In-memory specialist model cache ─────────────────────────────────
_mag_v2_models: dict = {}  # { "baseline:{ticker}": trained_model, ... }
_mag_v2_calibrators: dict = {}  # { "baseline": MagnitudeCalibrator, ... }


def _get_supabase_client():
    """Lazy import to avoid circular deps."""
    from supabase_client import get_client
    return get_client()


@app.post("/magnitude-v2/train")
async def magnitude_v2_train(
    req: MagnitudeV2TrainRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """
    Train magnitude v2 specialist models.
    Trains baseline, earnings, and/or event specialists per ticker.
    Results stored in magnitude_v2_specialist_outputs + calibration tables.
    """
    verify_caller(authorization)
    log.info(f"Magnitude v2 train: specialists={req.specialist_types}, "
             f"tickers={req.tickers or 'all'}, user={req.user_id[:8]}...")

    try:
        sb = _get_supabase_client()

        # Resolve tickers
        tickers = req.tickers
        if not tickers:
            # Fetch all tracked tickers from stock_impact_profiles
            resp = sb.table("stock_impact_profiles") \
                .select("ticker") \
                .eq("user_id", req.user_id) \
                .execute()
            tickers = list(set(r["ticker"] for r in resp.data)) if resp.data else []

        if not tickers:
            return {"status": "no_tickers", "success": False}

        results = {"status": "trained", "success": True, "tickers": {}, "summary": {}}
        total_trained = 0
        total_skipped = 0

        for ticker in tickers:
            ticker = ticker.upper()
            ticker_result = {"specialists": {}}

            # Fetch score_deltas for feature building
            raw = fetch_score_deltas(ticker, req.user_id, req.lookback_days)
            if len(raw) < MIN_SAMPLES:
                ticker_result["status"] = "insufficient_data"
                ticker_result["rows"] = len(raw)
                results["tickers"][ticker] = ticker_result
                total_skipped += 1
                continue

            # Build training matrix from score_deltas (not extract_features, which is prediction-only)
            X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
            if X is None or len(X) < MIN_SAMPLES:
                ticker_result["status"] = "insufficient_data"
                ticker_result["rows"] = len(raw)
                results["tickers"][ticker] = ticker_result
                total_skipped += 1
                continue

            y_signed = y
            y_abs = np.abs(y)
            feature_names = factor_names


            # Train each requested specialist
            if "baseline" in req.specialist_types:
                try:
                    baseline_result = train_baseline_model(
                        X, y_signed, y_abs, dates, feature_names,
                        purge_days=req.purge_days,
                        embargo_days=req.embargo_days,
                    )
                    _mag_v2_models[f"baseline:{ticker}"] = baseline_result
                    ticker_result["specialists"]["baseline"] = {
                        k: v for k, v in baseline_result.items() if k != "model"
                    }
                    total_trained += 1

                    # v15e: Persist model to Supabase Storage
                    try:
                        version = f"v2.0-{datetime.utcnow().strftime('%Y%m%d%H%M')}"
                        save_result = save_model_artifact(sb, "baseline", ticker, baseline_result, version, req.user_id)
                        ticker_result["specialists"]["baseline"]["model_stored"] = True
                        ticker_result["specialists"]["baseline"]["model_version"] = version
                        ticker_result["specialists"]["baseline"]["storage_path"] = save_result.get("path")
                    except Exception as store_err:
                        log.warning(f"Model storage failed for baseline:{ticker}: {store_err}")
                        ticker_result["specialists"]["baseline"]["model_stored"] = False
                except Exception as e:
                    log.warning(f"Baseline train failed for {ticker}: {e}")
                    ticker_result["specialists"]["baseline"] = {"status": "error", "error": str(e)}

            if "earnings" in req.specialist_types:
                try:
                    earnings_result = train_earnings_model(
                        X, y_signed, y_abs, dates, feature_names,
                        raw_data=raw,
                        purge_days=req.purge_days + 2,  # wider purge for earnings
                        embargo_days=req.embargo_days + 1,
                    )
                    _mag_v2_models[f"earnings:{ticker}"] = earnings_result
                    ticker_result["specialists"]["earnings"] = {
                        k: v for k, v in earnings_result.items() if k != "model"
                    }
                    total_trained += 1

                    # v15e: Persist model to Supabase Storage
                    try:
                        version = f"v2.0-{datetime.utcnow().strftime('%Y%m%d%H%M')}"
                        save_model_artifact(sb, "earnings", ticker, earnings_result, version, req.user_id)
                        ticker_result["specialists"]["earnings"]["model_stored"] = True
                        ticker_result["specialists"]["earnings"]["model_version"] = version
                    except Exception as store_err:
                        log.warning(f"Model storage failed for earnings:{ticker}: {store_err}")
                        ticker_result["specialists"]["earnings"]["model_stored"] = False
                except Exception as e:
                    log.warning(f"Earnings train failed for {ticker}: {e}")
                    ticker_result["specialists"]["earnings"] = {"status": "error", "error": str(e)}

            if "event" in req.specialist_types:
                try:
                    event_result = train_event_model(
                        X, y_signed, y_abs, dates, feature_names,
                        raw_data=raw,
                        purge_days=req.purge_days,
                        embargo_days=req.embargo_days,
                    )
                    _mag_v2_models[f"event:{ticker}"] = event_result
                    ticker_result["specialists"]["event"] = {
                        k: v for k, v in event_result.items() if k != "model"
                    }
                    total_trained += 1

                    # v15e: Persist model to Supabase Storage
                    try:
                        version = f"v2.0-{datetime.utcnow().strftime('%Y%m%d%H%M')}"
                        save_model_artifact(sb, "event", ticker, event_result, version, req.user_id)
                        ticker_result["specialists"]["event"]["model_stored"] = True
                        ticker_result["specialists"]["event"]["model_version"] = version
                    except Exception as store_err:
                        log.warning(f"Model storage failed for event:{ticker}: {store_err}")
                        ticker_result["specialists"]["event"]["model_stored"] = False
                except Exception as e:
                    log.warning(f"Event train failed for {ticker}: {e}")
                    ticker_result["specialists"]["event"] = {"status": "error", "error": str(e)}

            ticker_result["status"] = "trained"
            results["tickers"][ticker] = ticker_result

        results["summary"] = {
            "total_tickers": len(tickers),
            "trained": total_trained,
            "skipped": total_skipped,
            "feature_version": MAG_FEATURE_VERSION,
        }

        # Send callback if provided
        if req.callback_url:
            await _send_callback(req.callback_url, {
                **results,
                "model_type": "magnitude_v2",
                "user_id": req.user_id,
            })
            return {"accepted": True, "tickers": len(tickers)}

        return results

    except Exception as e:
        log.exception("Magnitude v2 train failed")
        error_result = {
            "success": False,
            "error": str(e),
            "user_id": req.user_id,
            "model_type": "magnitude_v2",
        }
        if req.callback_url:
            await _send_callback(req.callback_url, error_result)
            return {"accepted": True, "status": "error"}
        raise HTTPException(500, str(e))


@app.post("/magnitude-v2/predict")
async def magnitude_v2_predict(
    req: MagnitudeV2PredictRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Predict magnitude using v2 specialist ensemble.
    Returns: expected move, quantiles (q10-q90), path class, threshold probs,
    disagreement metrics, certainty level.
    All predictions stored with is_shadow=TRUE until v2 is validated.
    """
    verify_caller(authorization)
    ticker = req.ticker.upper()

    try:
        sb = _get_supabase_client()

        # Get latest features for this ticker
        raw = fetch_score_deltas(ticker, req.user_id, 30)
        if not raw:
            return {"status": "no_data", "ticker": ticker, "success": False}

        # ── Build inputs for extract_features() ──
        sector = get_sector(ticker)

        # Market context
        ff_resp = sb.table("fast_features") \
            .select("vix_level, features") \
            .eq("user_id", req.user_id) \
            .order("snapshot_at", desc=True) \
            .limit(1) \
            .execute()
        ff = ff_resp.data[0] if ff_resp.data else {}
        ff_feat = ff.get("features", {}) if isinstance(ff.get("features"), dict) else {}

        spy_resp = sb.table("live_ticks") \
            .select("price") \
            .eq("ticker", "SPY") \
            .order("received_at", desc=True) \
            .limit(10) \
            .execute()
        spy_prices = [r["price"] for r in reversed(spy_resp.data or [])]

        market_data = {
            "vix": ff.get("vix_level", 20),
            "vix_3m": ff_feat.get("vix_3m", ff.get("vix_level", 20)),
            "credit_spread": ff_feat.get("credit_spread", 0),
            "dxy": ff_feat.get("dollar_index", 0),
            "spy_prices": spy_prices,
        }

        # Stock data
        fund_resp = sb.table("fundamentals_cache") \
            .select("beta, implied_volatility, short_percent_float, short_ratio") \
            .eq("ticker", ticker) \
            .eq("user_id", req.user_id) \
            .order("fetched_at", desc=True) \
            .limit(1) \
            .execute()
        fund = fund_resp.data[0] if fund_resp.data else {}

        stock_data = {
            "prices": [r.get("actual_move_3d", 0) for r in raw if r.get("actual_move_3d") is not None],
            "volumes": [],
            "opens": [],
            "closes": [],
            "rsi14": 50,
            "bb_position": 0.5,
            "atr_pct": 1.0,
            "beta": fund.get("beta", 1.0),
        }

        # Sector data
        sector_data = {
            "sector": sector or "Unknown",
            "etf_prices": [],
            "dispersion": 0,
        }

        # Event data
        event_resp = sb.table("events") \
            .select("magnitude, direction, event_type") \
            .eq("ticker", ticker) \
            .eq("user_id", req.user_id) \
            .gte("created_at", (datetime.utcnow() - timedelta(days=7)).isoformat()) \
            .execute()
        events = event_resp.data or []

        event_data = {
            "has_earnings": any(e.get("event_type") == "earnings" for e in events),
            "days_to_earnings": None,
            "crowding": len(events),
            "catalyst_count_7d": len(events),
            "earnings_density": 0,
        }

        # Options & fundamentals
        options_data = {
            "implied_vol": fund.get("implied_volatility", 0),
            "skew": 0,
            "term_slope": 0,
            "put_call_ratio": 0,
            "implied_move": 0,
        }

        fundamentals = {
            "short_interest_ratio": fund.get("short_ratio", 0),
            "short_interest_change": 0,
        }

        features = mag_extract_features(
            ticker, market_data, stock_data, sector_data, event_data,
            options_data, fundamentals,
        )
        if not features:
            return {"status": "feature_extraction_failed", "ticker": ticker, "success": False}

        # v15e: Load from storage if not in memory (stateless container fix)
        def _ensure_model(specialist_type: str, t: str):
            """Check memory first, then storage. Cache after download."""
            key = f"{specialist_type}:{t}"
            model = _mag_v2_models.get(key)
            if model:
                return model, "memory"
            # Attempt storage fallback
            try:
                log.info(f"Attempting storage load for {specialist_type}:{t}")
                model = load_model_artifact(sb, specialist_type, t, req.user_id)
                if model:
                    _mag_v2_models[key] = model
                    log.info(f"Storage load SUCCESS for {specialist_type}:{t}, keys={list(model.keys())}")
                    return model, "storage"
                else:
                    log.warning(f"Storage load returned None for {specialist_type}:{t}")
            except Exception as e:
                log.warning(f"Storage load failed for {specialist_type}:{t}: {e}")
            return None, "cold_start"

        model_sources: dict[str, str] = {}

        # extract_features returns a flat dict, not a matrix — build X from feature names
        baseline_state, baseline_src = _ensure_model("baseline", ticker)
        model_sources["baseline"] = baseline_src
        feat_key = "features_used" if "features_used" in (baseline_state or {}) else "feature_names"
        if not isinstance(baseline_state, dict) or feat_key not in baseline_state or not baseline_state[feat_key]:
            return {
                "status": "no_trained_model",
                "ticker": ticker,
                "success": False,
                "model_sources": model_sources,
                "hint": "Run /magnitude-v2/train first to establish feature schema, or check storage persistence",
            }
        feature_names = baseline_state[feat_key]

        X_latest = np.array([[features.get(name, 0) or 0 for name in feature_names]], dtype=float)


        # Collect specialist predictions
        specialists: list[SpecialistOutput] = []

        # Baseline specialist (already loaded above)
        if baseline_state:
            try:
                baseline_pred = predict_baseline(
                    baseline_state,
                    X_latest[0],
                    baseline_state["point_model"],
                    baseline_state["quantile_models"],
                )

                specialists.append(baseline_pred)
            except Exception as e:
                log.warning(f"Baseline predict failed for {ticker}: {e}")

        # Earnings specialist (only if near earnings)
        earnings_model, earnings_src = _ensure_model("earnings", ticker)
        model_sources["earnings"] = earnings_src
        if earnings_model:
            try:
                from magnitude_v2.earnings_model import predict_earnings
                earnings_pred = predict_earnings(earnings_model, X_latest)
                specialists.append(earnings_pred)
            except Exception as e:
                log.warning(f"Earnings predict failed for {ticker}: {e}")

        # Event specialist
        event_model, event_src = _ensure_model("event", ticker)
        model_sources["event"] = event_src
        if event_model:
            try:
                from magnitude_v2.event_model import predict_event
                event_pred = predict_event(event_model, X_latest)
                specialists.append(event_pred)
            except Exception as e:
                log.warning(f"Event predict failed for {ticker}: {e}")

        if not specialists:
            return {
                "status": "no_specialists_available",
                "ticker": ticker,
                "success": False,
                "model_sources": model_sources,
                "hint": "Run /magnitude-v2/train first or check storage persistence",
            }

        # Fetch calibration metrics
        cal_resp = sb.table("magnitude_v2_calibration") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .execute()
        cal_metrics = {}
        if cal_resp.data:
            for row in cal_resp.data:
                cal_metrics[row["specialist_type"]] = {
                    "mae": row.get("mae", 999),
                    "rmse": row.get("rmse", 999),
                    "signed_bias": row.get("signed_bias", 0),
                }

        # Detect event context
        has_earnings = any(s.specialist_type == "earnings" for s in specialists)
        has_catalyst = any(s.specialist_type == "event" for s in specialists)

        # Compute dynamic weights
        weights = compute_dynamic_weights(
            specialists, cal_metrics,
            regime_state="unknown",  # TODO: pull from regime classifier
            has_earnings=has_earnings,
            has_catalyst=has_catalyst,
        )

        # Compute disagreement
        disagreement = compute_disagreement(specialists)

        # Blend predictions
        prediction = blend_predictions(
            specialists, weights, disagreement,
            regime_state="unknown",
            vol_regime="normal",
        )

        # Store as shadow prediction
        prediction["ticker"] = ticker
        prediction["user_id"] = req.user_id
        prediction["is_shadow"] = True
        prediction["horizon_days"] = req.horizon_days
        prediction["feature_version"] = MAG_FEATURE_VERSION
        prediction["success"] = True
        prediction["model_sources"] = model_sources  # v15e: memory/storage/cold_start per specialist

        # Write to magnitude_v2_predictions
        try:
            sb.table("magnitude_v2_predictions").insert({
                "user_id": req.user_id,
                "ticker": ticker,
                "horizon_days": req.horizon_days,
                "expected_move_pct": prediction["expected_move_pct"],
                "expected_abs_move_pct": prediction["expected_abs_move_pct"],
                "ci_low": prediction["ci_low"],
                "ci_high": prediction["ci_high"],
                "q10": prediction.get("q10", 0),
                "q25": prediction.get("q25", 0),
                "q50": prediction.get("q50", 0),
                "q75": prediction.get("q75", 0),
                "q90": prediction.get("q90", 0),
                "path_class": prediction.get("path_class", "drift"),
                "path_confidence": prediction.get("path_confidence", 0),
                "certainty_level": prediction.get("certainty_level", "low"),
                "meta_confidence": prediction.get("meta_confidence", 0),
                "specialist_weights": prediction.get("specialist_weights", {}),
                "winning_specialist": prediction.get("winning_specialist"),
                "specialist_disagreement": prediction.get("specialist_disagreement", 0),
                "disagreement_action": prediction.get("disagreement_action", "none"),
                "is_shadow": True,
                "feature_version": MAG_FEATURE_VERSION,
                "regime_state": prediction.get("regime_state", "unknown"),
                "vol_regime": prediction.get("vol_regime", "normal"),
            }).execute()
        except Exception as e:
            log.warning(f"Failed to store v2 prediction: {e}")

        return prediction

    except Exception as e:
        log.exception(f"Magnitude v2 predict failed for {ticker}")
        raise HTTPException(500, str(e))


@app.post("/magnitude-v2/calibrate")
async def magnitude_v2_calibrate(
    req: MagnitudeV2CalibrateRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Run calibration for a specialist type using resolved predictions.
    Updates magnitude_v2_calibration table with MAE, RMSE, CI coverage, bias.
    """
    verify_caller(authorization)
    log.info(f"Magnitude v2 calibrate: type={req.specialist_type}, user={req.user_id[:8]}...")

    try:
        sb = _get_supabase_client()

        # Fetch resolved predictions for this specialist
        resp = sb.table("magnitude_v2_predictions") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .eq("winning_specialist", req.specialist_type) \
            .not_.is_("actual_move_pct", "null") \
            .order("predicted_at", desc=True) \
            .limit(500) \
            .execute()

        if not resp.data or len(resp.data) < 20:
            return {
                "status": "insufficient_resolved",
                "samples": len(resp.data) if resp.data else 0,
                "min_required": 20,
            }

        predictions = [r["expected_move_pct"] for r in resp.data]
        actuals = [r["actual_move_pct"] for r in resp.data]
        ci_lows = [r["ci_low"] for r in resp.data]
        ci_highs = [r["ci_high"] for r in resp.data]

        calibrator = MagnitudeCalibrator()
        cal_result = calibrator.calibrate(predictions, actuals, ci_lows, ci_highs)

        _mag_v2_calibrators[req.specialist_type] = calibrator

        # Upsert calibration metrics
        sb.table("magnitude_v2_calibration").upsert({
            "user_id": req.user_id,
            "specialist_type": req.specialist_type,
            "subgroup_key": "global",
            "mae": cal_result["mae"],
            "rmse": cal_result["rmse"],
            "signed_bias": cal_result["signed_bias"],
            "ci_coverage_80": cal_result.get("ci_coverage_80"),
            "ci_coverage_90": cal_result.get("ci_coverage_90"),
            "total_samples": len(predictions),
            "recent_samples_30d": cal_result.get("recent_samples_30d", 0),
            "is_trusted": cal_result.get("is_trusted", False),
            "trust_reason": cal_result.get("trust_reason"),
            "calibration_method": "isotonic_conformal",
        }, on_conflict="user_id,specialist_type,subgroup_key").execute()

        return {
            "status": "calibrated",
            "specialist_type": req.specialist_type,
            "success": True,
            **cal_result,
        }

    except Exception as e:
        log.exception("Magnitude v2 calibrate failed")
        raise HTTPException(500, str(e))


@app.get("/magnitude-v2/health")
async def magnitude_v2_health():
    """Health check for magnitude v2 subsystem."""
    return {
        "status": "ok",
        "loaded_models_memory": list(_mag_v2_models.keys()),
        "loaded_calibrators": list(_mag_v2_calibrators.keys()),
        "feature_version": MAG_FEATURE_VERSION,
        "mode": "shadow",
    }
