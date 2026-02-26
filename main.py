"""
Stock ML Backend — FastAPI service on Railway.
Queries score_deltas from Supabase, trains LightGBM per-factor models,
writes optimized weights back to stock_impact_profiles + calibration_state.
"""
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from supabase_client import (
    fetch_score_deltas,
    fetch_score_deltas_range,
    fetch_current_weights,
    fetch_market_defaults,
    fetch_calibration_state,
    write_optimized_weights,
)

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

# ── Sector map (mirrors edge functions) ──────────────────────────────────
SECTOR_MAP: dict[str, str] = {}

def _sm(sector: str, tickers: list[str]):
    for t in tickers:
        SECTOR_MAP[t] = sector

_sm("Technology", ["AAPL","MSFT","GOOGL","GOOG","NVDA","META","AVGO","ORCL","CRM","ADBE","AMD","CSCO","ACN","INTC","IBM","INTU","NOW","QCOM","TXN","AMAT","ADI","LRCX","MU","KLAC","SNPS","CDNS","MRVL","FTNT","PANW","CRWD","NXPI","ON","MPWR","ANSS","KEYS","GEN","CTSH","IT","FSLR","EPAM","ENPH","SEDG","AKAM","FFIV","JNPR","SWKS","QRVO","WDC","PLTR","SOFI","HOOD","SNAP","PINS","ZM","ROKU","TWLO","SQ","COIN","AFRM","UPST","SHOP","NET","SNOW","DDOG","MDB","ZS","OKTA","S","MNDY","TEAM","ATLSY","TTD","TRADE","PATH","AI","BBAI","BIGC","CFLT","GTLB","HUBS","PCOR","ESTC","TYL","TOST","DLO","FOUR","GFS","WOLF","SMCI","ARM","VRT","GEV","IONQ","RGTI","QUBT","KULR","ALAB","OKLO","SMR","APLD","RMBS","SMTC","ONTO","GLOB","LSCC","PI","CGNX","MASI","CYBR","MANH","BILL","HQY","TNET","PAYC","CALX","FORM","DIOD","POWI","COHU","ICHR","AEIS","CSGS","HLIT","AMBA","BRZE","DOCN","VERX","SPSC","VRNT","PRGS","DUOL"])
_sm("Healthcare", ["UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","PFE","DHR","AMGN","ELV","BMY","ISRG","GILD","MDT","CI","SYK","VRTX","REGN","ZTS","BSX","BDX","HUM","IDXX","IQV","EW","A","DXCM","MTD","WAT","BAX","ALGN","HOLX","TFX","TECH","MOH","CNC","HSIC","XRAY","OGN","DVA","VTRS","CTLT","INCY","MRNA","BIIB","ILMN","PKI","LH","DGX","PCVX","NBIX","LNTH","BPMC","ALNY","HALO","RGEN","ENSG","CHE","EHC","EXEL","MEDP","GMED","NVST","MMSI","HAE","CPRX","TMDX","AGIO","XNCR","SMMT","RXRX","TEM","PDCO","OMCL","CORT"])
_sm("Financials", ["JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","SPGI","CME","ICE","MCO","AON","MMC","CB","PGR","MET","AIG","TRV","AFL","PRU","ALL","CINF","FITB","MTB","HBAN","RF","KEY","CFG","ZION","CMA","NDAQ","MSCI","CBOE","RJF","FRC","BEN","TROW","IVZ","WRB","GL","RE","L","AIZ","LNC","MKTX","OWL","IBKR","SF","EWBC","GBCI","FHN","WTFC","UMBF","SBCF","PB","WAL","BOH","CADE","CVBF","FNB","HWC","OZK","SSB","SNV","ASB","BKU","CUB","FBP","FCNCA","FFIN","HOPE","PIPR","PNFP","HLI","SCI","RNR","DKNG","SIGI","NMIH","TBBK","TCBI","WAFD","WSBC","WSFS","BANR","COLB","DCOM","FFBC","FULT","GABC","HAFC","HTLF","IBTX","INDB","MBWM","NBTB","NWBI","ONB","PEBO","PFBC","PPBI","RNST","SBSI","SFNC","TRMK","UBSI","UCBI","VBTX","WABC","EFSC","NBHC","OFG","NU","PAYO"])
_sm("Consumer Discretionary", ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG","ABNB","MAR","HLT","ORLY","AZO","ROST","DHI","LEN","PHM","NVR","GPC","POOL","BBY","APTV","GRMN","EBAY","ETSY","DRI","YUM","MGM","WYNN","CZR","LVS","RCL","CCL","NCLH","F","GM","TSCO","KMX","AAP","BWA","SEE","HAS","NWL","WHR","PVH","RL","TPR","VFC","PENN","DECK","WSM","LULU","BURL","FND","SFM","CROX","SKX","RIVN","LCID","DASH","RBLX","U","MELI","SE","GRAB","CPNG","BABA","JD","PDD","BIDU","NIO","XPEV","LI","CART","BIRK","CAVA","BOOT","CARG","SHOO","FIZZ","GMS"])
_sm("Consumer Staples", ["PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB","EL","STZ","GIS","SJM","K","HSY","CPB","HRL","MKC","CHD","CAG","TSN","KHC","BG","ADM","TAP","MNST","KDP","CLX","SYY","CELH","LANC"])
_sm("Industrials", ["CAT","DE","UNP","UPS","HON","GE","BA","RTX","LMT","MMM","GD","NOC","TT","PH","ROK","EMR","ETN","ITW","FAST","CTAS","PCAR","ODFL","CSX","NSC","FDX","DAL","UAL","LUV","ALK","JBHT","XYL","WM","RSG","VRSK","IR","DOV","SWK","SNA","RHI","PWR","SAIA","WFRD","RBC","FLR","JBL","KBR","CACI","AZEK","CSWI","ESAB","WTS","SITE","TTC","AOS","GNRC","TREX","UFPI","WEX","EXPO","AVNT","KTOS","INST","AIT","WOR","RXO","AAON","JOBY","ACHR","LILM","EVTL","BLDE","RKLB","ASTS","SPCE","LUNR","MNTS","RDW","GRRR"])
_sm("Energy", ["XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","PXD","DVN","OXY","HAL","FANG","HES","BKR","TRGP","WMB","KMI","OKE","CTRA"])
_sm("Utilities", ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL","WEC","ES","ED","PEG","AWK","ATO","CMS","CNP","NI","EVRG","FE","PPL","MGEE","CWEN"])
_sm("Real Estate", ["PLD","AMT","CCI","EQIX","PSA","DLR","O","WELL","SPG","AVB","EQR","VTR","ARE","MAA","UDR","ESS","PEAK","KIM","REG","BXP","COOP"])
_sm("Materials", ["LIN","APD","SHW","ECL","FCX","NEM","NUE","VMC","MLM","DOW","DD","PPG","CE","ALB","FMC","CF","MOS","IFF","EMN","WDFC","BCPC"])
_sm("Communication Services", ["DIS","NFLX","CMCSA","T","VZ","CHTR","TMUS","EA","TTWO","WBD","PARA","FOX","FOXA","NWS","NWSA","MTCH","IPG","OMC","LYV","BRBR"])
_sm("Crypto/Blockchain", ["IREN","CLSK","MARA","RIOT","HUT","BITF","WULF","CIFR","CORZ"])

def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "unknown")


# ── Auth helper ──────────────────────────────────────────────────────────
def verify_caller(authorization: Optional[str] = Header(None)):
    if API_SECRET and authorization != f"Bearer {API_SECRET}":
        raise HTTPException(401, "Unauthorized")


# ── Request models ───────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker: str
    user_id: str
    lookback_days: int = 365


class BacktestRequest(BaseModel):
    ticker: str
    user_id: str
    start_date: str
    end_date: str


# ── Feature engineering: pivot score_deltas by event_type ────────────────
MIN_SAMPLES = 15          # ← lowered from 30 for early-stage data collection
MAX_WEIGHT = 4.0


def build_factor_matrix(raw_deltas: list[dict], target_col: str = "actual_move_3d"):
    """
    Pivot decomposed per-factor score_deltas into a feature matrix.

    Each row = one (measured_date, ticker) observation.
    Each column = one event_type's weighted contribution (magnitude × weight × decay).
    Target = actual stock move over the horizon.
    """
    df = pd.DataFrame(raw_deltas)
    if df.empty:
        return None, None, []

    # Compute each factor's contribution
    df["contribution"] = (
        df["predicted_impact"].fillna(0)
        * df["weight_used"].fillna(1)
        * df["decay_at_measurement"].fillna(1)
    )

    # Use measured_date as the grouping key (one observation per day)
    if "measured_date" not in df.columns or df["measured_date"].isna().all():
        df["measured_date"] = pd.to_datetime(df["measured_at"]).dt.date

    # Pivot: rows=dates, columns=event_types, values=sum of contributions
    pivot = df.pivot_table(
        index="measured_date",
        columns="event_type",
        values="contribution",
        aggfunc="sum",
        fill_value=0,
    )

    # Get target: one actual_move per date (they should all be the same for a date)
    targets = (
        df.groupby("measured_date")[target_col]
        .first()
        .reindex(pivot.index)
    )

    # Drop rows where target is NaN
    valid = targets.notna()
    pivot = pivot[valid]
    targets = targets[valid]

    if len(pivot) < MIN_SAMPLES:
        return None, None, []

    # Sort chronologically
    pivot = pivot.sort_index()
    targets = targets.reindex(pivot.index)

    factor_names = pivot.columns.tolist()
    return pivot.values, targets.values, factor_names


def importances_to_weights(
    importances: np.ndarray,
    factor_names: list[str],
    market_defaults: dict[str, float],
    current_weights: dict[str, dict],
) -> dict[str, float]:
    """
    Convert LightGBM feature importances to weight overrides.

    Strategy: scale importances so the most important factor gets a weight
    proportional to its baseline, capped at MAX_WEIGHT. Less important
    factors get proportionally smaller weights.
    """
    if len(importances) == 0:
        return {}

    # Normalize importances to [0, 1]
    total = importances.sum()
    if total == 0:
        return {f: market_defaults.get(f, 1.0) for f in factor_names}

    normed = importances / total
    weights = {}

    for i, factor in enumerate(factor_names):
        baseline = market_defaults.get(factor, 1.0)
        current = current_weights.get(factor, {}).get("weight_override")

        # Scale: importance-weighted blend between baseline and MAX_WEIGHT
        # High importance → weight closer to MAX_WEIGHT
        # Low importance → weight closer to 0
        raw_weight = baseline + normed[i] * (MAX_WEIGHT - baseline) * 2
        raw_weight = np.clip(raw_weight, -MAX_WEIGHT, MAX_WEIGHT)

        # Smooth with current weight if it exists (avoid wild jumps)
        if current is not None:
            raw_weight = 0.6 * raw_weight + 0.4 * current

        weights[factor] = round(float(raw_weight), 4)

    return weights


# ── Endpoints ────────────────────────────────────────────────────────────
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

        X, y, factor_names = build_factor_matrix(raw, "actual_move_3d")
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

        # ── 3. Train LightGBM with walk-forward validation ──
        n_splits = min(5, max(2, len(X) // 15))   # ← adjusted for lower threshold
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        cv_correlations = []
        best_model = None
        best_mse = float("inf")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=max(3, len(train_idx) // 20),  # ← lowered from 5
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

        # Direction accuracy
        direction_correct = np.sum(np.sign(full_preds) == np.sign(y))
        direction_accuracy = float(direction_correct / len(y)) if len(y) > 0 else 0

        log.info(
            f"  Training complete: corr={correlation:.3f}, "
            f"dir_acc={direction_accuracy:.1%}, "
            f"cv_mse={cv_scores}, factors={len(factor_names)}"
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
        }

    except Exception as e:
        log.exception(f"Train failed for {ticker}")
        raise HTTPException(500, str(e))


@app.post("/backtest")
async def backtest(req: BacktestRequest, authorization: Optional[str] = Header(None)):
    verify_caller(authorization)
    ticker = req.ticker.upper()
    log.info(f"Backtest request: {ticker}, {req.start_date} → {req.end_date}")

    try:
        # ── 1. Fetch data ──
        raw = fetch_score_deltas_range(ticker, req.user_id, req.start_date, req.end_date)
        log.info(f"  Fetched {len(raw)} score_delta rows for {ticker}")

        if len(raw) < 10:       # ← lowered from 20 for early-stage testing
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": 10,
            }

        X, y, factor_names = build_factor_matrix(raw, "actual_move_3d")
        if X is None:
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "min_required": MIN_SAMPLES,
            }

        # ── 2. Walk-forward 70/30 split ──
        split = int(len(X) * 0.7)
        if split < 8 or (len(X) - split) < 5:    # ← lowered from 15/10
            return {
                "status": "insufficient_data",
                "rows": len(raw),
                "observations": len(X),
                "reason": "Not enough data for 70/30 split",
            }

        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        log.info(f"  Split: {len(X_train)} train / {len(X_test)} test, {len(factor_names)} factors")

        # ── 3. Train ──
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X_train) // 20),  # ← lowered from 5
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ── 4. Metrics ──
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

        # Per-factor accuracy breakdown
        factor_performance = {}
        for i, factor in enumerate(factor_names):
            col = X_test[:, i]
            nonzero = col != 0
            if nonzero.sum() > 3:      # ← lowered from 5
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
            "samples_train": split,
            "samples_test": len(X) - split,
            "factors": factor_names,
            "mse": round(mse, 6),
            "correlation": round(correlation, 4),
            "direction_accuracy": round(direction_accuracy, 4),
            "importance": importance_dict,
            "factor_performance": factor_performance,
        }

    except Exception as e:
        log.exception(f"Backtest failed for {ticker}")
        raise HTTPException(500, str(e))
