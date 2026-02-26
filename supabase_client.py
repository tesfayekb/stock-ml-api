"""
Supabase data access layer for the Stock ML Backend.
Uses service_role key to bypass RLS, filtering by user_id explicitly.
"""
import os
from datetime import datetime, timedelta
from supabase import create_client, Client

_client: Client | None = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _client


def fetch_score_deltas(
    ticker: str, user_id: str, lookback_days: int = 365, limit: int = 5000
) -> list[dict]:
    """Fetch decomposed per-factor score_deltas for a ticker."""
    sb = get_supabase()
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    resp = (
        sb.table("score_deltas")
        .select(
            "id, ticker, event_type, tier, predicted_impact, weight_used, "
            "decay_at_measurement, actual_move_1d, actual_move_3d, actual_move_7d, "
            "measured_at, measured_date, regime_snapshot"
        )
        .eq("ticker", ticker)
        .eq("user_id", user_id)
        .gte("measured_at", cutoff)
        .order("measured_at")
        .limit(limit)
        .execute()
    )
    return resp.data or []


def fetch_score_deltas_range(
    ticker: str, user_id: str, start_date: str, end_date: str, limit: int = 5000
) -> list[dict]:
    """Fetch score_deltas within a date range for backtesting."""
    sb = get_supabase()
    resp = (
        sb.table("score_deltas")
        .select(
            "id, ticker, event_type, tier, predicted_impact, weight_used, "
            "decay_at_measurement, actual_move_1d, actual_move_3d, actual_move_7d, "
            "measured_at, measured_date, regime_snapshot"
        )
        .eq("ticker", ticker)
        .eq("user_id", user_id)
        .gte("measured_at", start_date)
        .lte("measured_at", end_date)
        .order("measured_at")
        .limit(limit)
        .execute()
    )
    return resp.data or []


def fetch_current_weights(ticker: str, user_id: str) -> dict[str, dict]:
    """Get current stock_impact_profiles for a ticker."""
    sb = get_supabase()
    resp = (
        sb.table("stock_impact_profiles")
        .select("event_type, weight_override, confidence, sample_size, sector")
        .eq("ticker", ticker)
        .eq("user_id", user_id)
        .execute()
    )
    return {r["event_type"]: r for r in (resp.data or [])}


def fetch_market_defaults(user_id: str) -> dict[str, float]:
    """Get baseline market_defaults weights."""
    sb = get_supabase()
    resp = (
        sb.table("market_defaults")
        .select("event_type, weight")
        .eq("user_id", user_id)
        .execute()
    )
    return {r["event_type"]: r["weight"] for r in (resp.data or [])}


def fetch_calibration_state(ticker: str, user_id: str) -> dict | None:
    """Get existing calibration state for champion-restore logic."""
    sb = get_supabase()
    resp = (
        sb.table("calibration_state")
        .select("*")
        .eq("ticker", ticker)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def write_optimized_weights(
    ticker: str,
    user_id: str,
    sector: str,
    weights: dict[str, float],
    correlation: float,
    sample_size: int,
    prev_best: float | None = None,
) -> dict:
    """Write LightGBM-optimized weights back to Supabase."""
    sb = get_supabase()
    now = datetime.utcnow().isoformat()

    # ── Write per-factor weights to stock_impact_profiles ──
    for event_type, weight in weights.items():
        sb.table("stock_impact_profiles").upsert(
            {
                "ticker": ticker,
                "user_id": user_id,
                "event_type": event_type,
                "sector": sector,
                "weight_override": round(weight, 4),
                "confidence": min(0.95, 0.3 + sample_size / 200),
                "sample_size": sample_size,
                "last_calibrated_at": now,
            },
            on_conflict="ticker,event_type,user_id",
        ).execute()

    # ── Update calibration_state ──
    is_new_best = prev_best is None or correlation > prev_best
    upsert_data = {
        "ticker": ticker,
        "user_id": user_id,
        "status": "calibrated",
        "last_calibrated_at": now,
        "last_correlation": round(correlation, 4),
        "last_weights": weights,
    }
    if is_new_best:
        upsert_data["best_correlation"] = round(correlation, 4)
        upsert_data["best_weights"] = weights

    sb.table("calibration_state").upsert(
        upsert_data, on_conflict="ticker,user_id"
    ).execute()

    return {"weights_written": len(weights), "is_new_best": is_new_best}
