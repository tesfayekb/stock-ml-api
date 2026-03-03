


"""
Stock ML Backend — FastAPI service on Railway.
Queries score_deltas from Supabase, trains LightGBM/Ridge/MLP per-factor models,
writes optimized weights back to stock_impact_profiles + calibration_state.
"""
import os
import logging
from typing import Optional

import httpx
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
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
            "rows": len(raw),
            "min_required": MIN_SAMPLES,
            "success": False,
        }

    X, y, factor_names, dates = build_factor_matrix(raw, "actual_move_3d")
    if X is None:
        return {
            "status": "insufficient_data",
            "ticker": ticker,
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
    return {"status": "ok"}


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
