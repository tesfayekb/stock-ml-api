"""
Feature engineering: pivot score_deltas into factor matrices,
extract enriched features from regime_snapshot metadata,
convert importances to weights, and compute SHAP attribution.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("ml-backend")

MIN_SAMPLES = 10
MAX_WEIGHT = 4.0

# Enriched feature names extracted from regime_snapshot
ENRICHED_FEATURES = [
    "rsi14",
    "momentum30d",
    "technical_score",
    "fundamental_score",
    "bb_position",
    "sector_relative_alpha",
]


def _extract_enriched(regime_snapshot: dict | None) -> dict[str, float]:
    """Extract enriched technical/fundamental features from regime_snapshot."""
    if not regime_snapshot or not isinstance(regime_snapshot, dict):
        return {}
    out = {}
    for feat in ENRICHED_FEATURES:
        val = regime_snapshot.get(feat)
        if val is not None:
            try:
                out[feat] = float(val)
            except (ValueError, TypeError):
                pass
    return out


def build_factor_matrix(raw_deltas: list[dict], target_col: str = "actual_move_3d"):
    """
    Pivot decomposed per-factor score_deltas into a feature matrix.

    Each row = one (measured_date, ticker) observation.
    Columns = event_type contributions + enriched technical/fundamental signals.
    Target = actual stock move over the horizon.

    Returns: (X, y, factor_names, dates) or (None, None, [], []) if insufficient data.
    """
    df = pd.DataFrame(raw_deltas)
    if df.empty:
        return None, None, [], []

    df["contribution"] = (
        df["predicted_impact"].fillna(0)
        * df["weight_used"].fillna(1)
        * df["decay_at_measurement"].fillna(1)
    )

    if "measured_date" not in df.columns or df["measured_date"].isna().all():
        df["measured_date"] = pd.to_datetime(df["measured_at"]).dt.date

    # ── Event-type pivot (original features) ──
    pivot = df.pivot_table(
        index="measured_date",
        columns="event_type",
        values="contribution",
        aggfunc="sum",
        fill_value=0,
    )

    # ── Extract enriched features from regime_snapshot ──
    enriched_rows = []
    for date, group in df.groupby("measured_date"):
        enriched = {}
        for _, row in group.iterrows():
            rs = row.get("regime_snapshot")
            if rs and isinstance(rs, dict):
                enriched = _extract_enriched(rs)
                break
        enriched["measured_date"] = date
        enriched_rows.append(enriched)

    enriched_df = pd.DataFrame(enriched_rows).set_index("measured_date")
    enriched_df = enriched_df.reindex(pivot.index)

    # Only include enriched features that have >50% non-null coverage
    enriched_cols = []
    for feat in ENRICHED_FEATURES:
        if feat in enriched_df.columns:
            coverage = enriched_df[feat].notna().mean()
            if coverage >= 0.5:
                enriched_cols.append(feat)
                col = enriched_df[feat].fillna(0)
                if feat == "rsi14":
                    col = (col - 50) / 50
                elif feat == "momentum30d":
                    col = col / 10
                elif feat in ("technical_score", "fundamental_score"):
                    col = col / 100
                elif feat == "bb_position":
                    col = col / 100
                elif feat == "sector_relative_alpha":
                    col = col / 5
                pivot[feat] = col

    if enriched_cols:
        log.info(f"  Enriched features added: {enriched_cols}")

    targets = (
        df.groupby("measured_date")[target_col]
        .first()
        .reindex(pivot.index)
    )

    valid = targets.notna()
    pivot = pivot[valid]
    targets = targets[valid]

    if len(pivot) < MIN_SAMPLES:
        return None, None, [], []

    pivot = pivot.sort_index()
    targets = targets.reindex(pivot.index)

    factor_names = pivot.columns.tolist()
    dates = [str(d) for d in pivot.index.tolist()]
    return pivot.values, targets.values, factor_names, dates


def importances_to_weights(
    importances: np.ndarray,
    factor_names: list[str],
    market_defaults: dict[str, float],
    current_weights: dict[str, dict],
) -> dict[str, float]:
    """
    Convert LightGBM feature importances to weight overrides.

    Strategy: scale importances so the most important factor gets a weight
    proportional to its baseline, capped at MAX_WEIGHT.
    Smooths with current weight (60/40 blend) to avoid wild jumps.
    """
    if len(importances) == 0:
        return {}

    total = importances.sum()
    if total == 0:
        return {f: market_defaults.get(f, 1.0) for f in factor_names}

    normed = importances / total
    weights = {}

    for i, factor in enumerate(factor_names):
        baseline = market_defaults.get(factor, 1.0)
        current = current_weights.get(factor, {}).get("weight_override")

        raw_weight = baseline + normed[i] * (MAX_WEIGHT - baseline) * 2
        raw_weight = np.clip(raw_weight, -MAX_WEIGHT, MAX_WEIGHT)

        if current is not None:
            raw_weight = 0.6 * raw_weight + 0.4 * current

        weights[factor] = round(float(raw_weight), 4)

    return weights


def compute_shap_importance(model, X_val, feature_names) -> dict:
    """
    Compute mean absolute SHAP values per feature using TreeExplainer.
    Returns dict of { feature_name: percentage }.
    Falls back to empty dict on error.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        total = mean_abs_shap.sum()

        importance = {}
        for fname, val in zip(feature_names, mean_abs_shap):
            importance[fname] = round(float(val / total * 100) if total > 0 else 0, 2)

        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return {}
