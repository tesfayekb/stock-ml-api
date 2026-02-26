import os, httpx, pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from typing import Optional
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

app = FastAPI(title="Stock ML Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
POLYGON_KEY = os.environ["POLYGON_API_KEY"]
API_SECRET = os.environ.get("ML_API_SECRET", "")

def verify_caller(authorization: Optional[str] = Header(None)):
    if API_SECRET and authorization != f"Bearer {API_SECRET}":
        raise HTTPException(401, "Unauthorized")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

class TrainRequest(BaseModel):
    ticker: str
    user_id: str
    lookback_days: int = 365

class BacktestRequest(BaseModel):
    ticker: str
    user_id: str
    start_date: str
    end_date: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
async def train(req: TrainRequest, authorization: Optional[str] = Header(None)):
    verify_caller(authorization)
    try:
        # Fetch score_deltas for this ticker
        res = sb.table("score_deltas").select("*").eq("ticker", req.ticker).eq("user_id", req.user_id).order("measured_at", desc=True).limit(500).execute()
        if not res.data or len(res.data) < 30:
            return {"status": "insufficient_data", "rows": len(res.data or [])}

        df = pd.DataFrame(res.data)
        features = ["predicted_impact", "weight_used", "decay_at_measurement"]
        target = "actual_move_3d"

        df = df.dropna(subset=features + [target])
        X, y = df[features].values, df[target].values

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        best_model = None
        best_score = float("inf")

        for train_idx, val_idx in tscv.split(X):
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            mse = mean_squared_error(y[val_idx], preds)
            scores.append(mse)
            if mse < best_score:
                best_score = mse
                best_model = model

        importance = dict(zip(features, best_model.feature_importances_.tolist()))
        correlation = float(np.corrcoef(best_model.predict(X), y)[0, 1])

        # Update calibration_state
        sb.table("calibration_state").upsert({
            "ticker": req.ticker,
            "user_id": req.user_id,
            "status": "calibrated",
            "last_calibrated_at": pd.Timestamp.now().isoformat(),
            "last_correlation": correlation,
            "best_correlation": correlation,
            "last_weights": importance,
            "best_weights": importance,
        }, on_conflict="ticker,user_id").execute()

        return {"status": "trained", "correlation": correlation, "cv_mse": scores, "importance": importance}

    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/backtest")
async def backtest(req: BacktestRequest, authorization: Optional[str] = Header(None)):
    verify_caller(authorization)
    try:
        res = sb.table("score_deltas").select("*").eq("ticker", req.ticker).eq("user_id", req.user_id).gte("measured_at", req.start_date).lte("measured_at", req.end_date).order("measured_at").execute()

        if not res.data or len(res.data) < 20:
            return {"status": "insufficient_data"}

        df = pd.DataFrame(res.data)
        features = ["predicted_impact", "weight_used", "decay_at_measurement"]
        target = "actual_move_3d"
        df = df.dropna(subset=features + [target])

        split = int(len(df) * 0.7)
        X_train, y_train = df[features].iloc[:split].values, df[target].iloc[:split].values
        X_test, y_test = df[features].iloc[split:].values, df[target].iloc[split:].values

        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "status": "complete",
            "mse": float(mean_squared_error(y_test, preds)),
            "correlation": float(np.corrcoef(preds, y_test)[0, 1]),
            "samples_train": split,
            "samples_test": len(df) - split,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
