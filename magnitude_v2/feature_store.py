"""
Magnitude v2 Feature Store — extracts and freezes versioned feature snapshots
for all specialist models. Features are grouped into market, volatility,
sector, stock, event, options, and residual categories.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

log = logging.getLogger("magnitude-v2")

FEATURE_VERSION = "v2.0"

# Feature groups with extraction logic
MARKET_FEATURES = [
    "vix_level", "vix_term_slope", "credit_spread", "dollar_index",
    "spy_return_1d", "spy_return_3d", "spy_return_5d",
]

VOL_FEATURES = [
    "realized_vol_3d", "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    "implied_vol", "iv_rv_spread", "vol_regime",
]

SECTOR_FEATURES = [
    "sector", "sector_relative_strength", "sector_dispersion",
    "sector_return_1d", "sector_return_3d",
]

STOCK_FEATURES = [
    "rsi14", "momentum_3d", "momentum_5d", "momentum_10d",
    "bb_position", "atr_pct", "volume_ratio_20d", "volume_persistence_3d",
    "gap_frequency_20d", "post_gap_followthrough",
]

EVENT_FEATURES = [
    "has_earnings_upcoming", "days_to_earnings", "earnings_density_sector",
    "event_crowding_score", "catalyst_count_7d",
]

OPTIONS_FEATURES = [
    "options_implied_move", "options_skew", "options_term_slope", "put_call_ratio",
]

RESIDUAL_FEATURES = [
    "market_beta", "residual_vs_market_3d", "residual_vs_sector_3d",
    "pre_event_drift_5d", "short_interest_ratio", "short_interest_change",
]


def compute_realized_vol(prices: list[float], window: int) -> float:
    """Annualized realized volatility from log returns."""
    if len(prices) < window + 1:
        return 0.0
    returns = np.diff(np.log(prices[-window - 1:]))
    return float(np.std(returns) * np.sqrt(252) * 100)


def compute_volume_persistence(volumes: list[float], window: int = 3) -> float:
    """Fraction of recent days where volume exceeded 20d average."""
    if len(volumes) < 20:
        return 0.0
    avg_20d = np.mean(volumes[-20:])
    if avg_20d == 0:
        return 0.0
    recent = volumes[-window:]
    return float(sum(1 for v in recent if v > avg_20d) / len(recent))


def compute_gap_frequency(opens: list[float], closes: list[float], window: int = 20) -> float:
    """Fraction of days in window with >0.5% gap from prior close."""
    if len(opens) < window or len(closes) < window:
        return 0.0
    gaps = 0
    for i in range(1, min(window, len(opens))):
        if closes[i - 1] > 0:
            gap_pct = abs((opens[i] - closes[i - 1]) / closes[i - 1]) * 100
            if gap_pct > 0.5:
                gaps += 1
    return float(gaps / (window - 1))


def compute_post_gap_followthrough(
    opens: list[float], closes: list[float], window: int = 20
) -> float:
    """Average signed follow-through after gaps. Positive = gap continues."""
    if len(opens) < window or len(closes) < window:
        return 0.0
    followthroughs = []
    for i in range(1, min(window, len(opens))):
        if closes[i - 1] > 0:
            gap_pct = (opens[i] - closes[i - 1]) / closes[i - 1] * 100
            if abs(gap_pct) > 0.5 and opens[i] > 0:
                day_move = (closes[i] - opens[i]) / opens[i] * 100
                # Positive if day move continues gap direction
                followthroughs.append(day_move * np.sign(gap_pct))
    return float(np.mean(followthroughs)) if followthroughs else 0.0


def extract_features(
    ticker: str,
    market_data: dict,
    stock_data: dict,
    sector_data: dict,
    event_data: dict,
    options_data: dict | None = None,
    fundamentals: dict | None = None,
) -> dict:
    """
    Extract frozen feature snapshot for magnitude v2 prediction.
    
    Args:
        ticker: Stock ticker
        market_data: { vix, vix_3m, credit_spread, dxy, spy_prices: [...] }
        stock_data: { prices: [...], volumes: [...], opens: [...], closes: [...],
                      rsi14, bb_position, atr_pct, beta }
        sector_data: { sector, etf_prices: [...], dispersion }
        event_data: { has_earnings, days_to_earnings, crowding, catalyst_count_7d,
                      earnings_density }
        options_data: { implied_move, skew, term_slope, put_call_ratio, implied_vol }
        fundamentals: { short_interest_ratio, short_interest_change }
    
    Returns:
        Feature dict matching magnitude_v2_features schema.
    """
    features = {
        "ticker": ticker,
        "feature_version": FEATURE_VERSION,
    }
    
    # ── Market context ──
    spy_prices = market_data.get("spy_prices", [])
    features["vix_level"] = market_data.get("vix", 0)
    vix_3m = market_data.get("vix_3m", 0)
    features["vix_term_slope"] = (
        (vix_3m - features["vix_level"]) / max(features["vix_level"], 1)
        if features["vix_level"] > 0 else 0
    )
    features["credit_spread"] = market_data.get("credit_spread", 0)
    features["dollar_index"] = market_data.get("dxy", 0)
    
    if len(spy_prices) >= 6:
        features["spy_return_1d"] = (spy_prices[-1] / spy_prices[-2] - 1) * 100
        features["spy_return_3d"] = (spy_prices[-1] / spy_prices[-4] - 1) * 100
        features["spy_return_5d"] = (spy_prices[-1] / spy_prices[-6] - 1) * 100
    
    # ── Volatility ──
    prices = stock_data.get("prices", [])
    features["realized_vol_3d"] = compute_realized_vol(prices, 3)
    features["realized_vol_5d"] = compute_realized_vol(prices, 5)
    features["realized_vol_10d"] = compute_realized_vol(prices, 10)
    features["realized_vol_20d"] = compute_realized_vol(prices, 20)
    
    iv = (options_data or {}).get("implied_vol", 0)
    features["implied_vol"] = iv
    rv20 = features["realized_vol_20d"]
    features["iv_rv_spread"] = iv - rv20 if iv and rv20 else 0
    
    # Vol regime classification
    if features["vix_level"] > 30:
        features["vol_regime"] = "crisis"
    elif features["vix_level"] > 22:
        features["vol_regime"] = "elevated"
    elif features["vix_level"] > 15:
        features["vol_regime"] = "normal"
    else:
        features["vol_regime"] = "compressed"
    
    # ── Sector ──
    features["sector"] = sector_data.get("sector", "unknown")
    etf_prices = sector_data.get("etf_prices", [])
    if len(etf_prices) >= 4 and len(prices) >= 4:
        stock_ret_3d = (prices[-1] / prices[-4] - 1) * 100 if prices[-4] > 0 else 0
        sector_ret_3d = (etf_prices[-1] / etf_prices[-4] - 1) * 100 if etf_prices[-4] > 0 else 0
        features["sector_relative_strength"] = stock_ret_3d - sector_ret_3d
        features["sector_return_3d"] = sector_ret_3d
    if len(etf_prices) >= 2:
        features["sector_return_1d"] = (etf_prices[-1] / etf_prices[-2] - 1) * 100 if etf_prices[-2] > 0 else 0
    features["sector_dispersion"] = sector_data.get("dispersion", 0)
    
    # ── Stock-specific ──
    features["rsi14"] = stock_data.get("rsi14", 50)
    if len(prices) >= 11:
        features["momentum_3d"] = (prices[-1] / prices[-4] - 1) * 100 if prices[-4] > 0 else 0
        features["momentum_5d"] = (prices[-1] / prices[-6] - 1) * 100 if prices[-6] > 0 else 0
        features["momentum_10d"] = (prices[-1] / prices[-11] - 1) * 100 if prices[-11] > 0 else 0
    features["bb_position"] = stock_data.get("bb_position", 50)
    features["atr_pct"] = stock_data.get("atr_pct", 0)
    
    volumes = stock_data.get("volumes", [])
    if len(volumes) >= 20:
        avg_20d = np.mean(volumes[-20:])
        features["volume_ratio_20d"] = float(volumes[-1] / avg_20d) if avg_20d > 0 else 1.0
    features["volume_persistence_3d"] = compute_volume_persistence(volumes)
    
    opens = stock_data.get("opens", [])
    closes = stock_data.get("closes", [])
    features["gap_frequency_20d"] = compute_gap_frequency(opens, closes)
    features["post_gap_followthrough"] = compute_post_gap_followthrough(opens, closes)
    
    # ── Event/catalyst ──
    features["has_earnings_upcoming"] = event_data.get("has_earnings", False)
    features["days_to_earnings"] = event_data.get("days_to_earnings")
    features["earnings_density_sector"] = event_data.get("earnings_density", 0)
    features["event_crowding_score"] = event_data.get("crowding", 0)
    features["catalyst_count_7d"] = event_data.get("catalyst_count_7d", 0)
    
    # ── Options-derived ──
    if options_data:
        features["options_implied_move"] = options_data.get("implied_move", 0)
        features["options_skew"] = options_data.get("skew", 0)
        features["options_term_slope"] = options_data.get("term_slope", 0)
        features["put_call_ratio"] = options_data.get("put_call_ratio", 0)
    
    # ── Residual / short interest ──
    features["market_beta"] = stock_data.get("beta", 1.0)
    if len(prices) >= 4 and len(spy_prices) >= 4:
        stock_ret = (prices[-1] / prices[-4] - 1) * 100 if prices[-4] > 0 else 0
        market_ret = (spy_prices[-1] / spy_prices[-4] - 1) * 100 if spy_prices[-4] > 0 else 0
        beta = features["market_beta"] or 1.0
        features["residual_vs_market_3d"] = stock_ret - beta * market_ret
        if "sector_return_3d" in features:
            features["residual_vs_sector_3d"] = stock_ret - features["sector_return_3d"]
    
    # Pre-event drift
    if len(prices) >= 6:
        features["pre_event_drift_5d"] = (prices[-1] / prices[-6] - 1) * 100 if prices[-6] > 0 else 0
    
    if fundamentals:
        features["short_interest_ratio"] = fundamentals.get("short_interest_ratio")
        features["short_interest_change"] = fundamentals.get("short_interest_change")
    
    # ── Data quality score ──
    total_fields = len(MARKET_FEATURES) + len(VOL_FEATURES) + len(STOCK_FEATURES) + len(EVENT_FEATURES)
    populated = sum(1 for k in features if features.get(k) is not None and k not in ("ticker", "feature_version"))
    features["data_quality_score"] = round(populated / max(total_fields, 1) * 100, 1)
    
    return features
