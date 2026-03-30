"""
Tactical 24h Forecast Model v2 — Canonical Schema + Decision Layer.

Changes from v1:
- FEATURE_KEYS aligned to fast-feature-poller canonical output
- v5 alpha features added (gamma_flip_distance, sweep_intensity, etc.)
- Binary direction + smart abstain (replaces 3-class up/flat/down)
- Opportunity filter (is there enough edge to act?)
- Ridge companion model for ensemble diversity
- Isotonic calibration placeholder
- Schema validation (fail loudly on missing critical features)
- Regime-conditioned confidence modulation

Model version: "tactical-24h-v2"
"""
import os
import logging
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from supabase import create_client
from datetime import datetime, timedelta

log = logging.getLogger("tactical-model-24h")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MIN_TACTICAL_SAMPLES = 200
MODEL_VERSION = "tactical-24h-v2"

# ══════════════════════════════════════════════════════════════════
# CANONICAL FEATURE SCHEMA — matches fast-feature-poller v5 output
# This is the SINGLE SOURCE OF TRUTH for feature names.
# If the poller changes a key, update HERE and retrain.
# ══════════════════════════════════════════════════════════════════

# Tier 1: Core macro context (always available, highest signal for 24h)
TIER1_MACRO = [
    "vix_level",              # was: vix_raw
    "vix_zscore_60d",         # was: vix_zscore
    "vix_percentile_252d",    # was: vix_percentile
    "vix_term_ratio",         # NEW: VIX vs VX1 (term structure)
    "vix_term_slope",         # NEW: VX2-VX1 slope (was: vix_acceleration proxy)
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "credit_spread_raw",
    "credit_spread_change_1d",   # was: credit_spread_change
    "credit_spread_zscore_60d",  # was: credit_spread_zscore
    "dgs2_level",
    "dgs2_change_1d",         # was: dgs2_change
    "yield_curve_slope",
]

# Tier 2: Cross-asset & calendar (high value for 24h regime context)
TIER2_CROSS_ASSET = [
    "days_to_fomc",
    "days_to_opex",
    "earnings_density_5d",    # was: earnings_density
    "iwm_spy_spread",
    "qqq_spy_spread",
    "hyg_tlt_spread_change",
    "dollar_return_1d",       # was: uup_return / dxy_momentum
    "gold_return_1d",         # was: gld_return
    "bond_return_1d",         # was: tlt_return
    "oil_return_1d",          # NEW (was unused)
    "btc_return_1d",          # NEW (was unused)
    "sector_dispersion",
]

# Tier 3: Intraday context (important for end-of-day 24h forecasts)
TIER3_INTRADAY = [
    "realized_vol_proxy_1h",  # was: realized_vol_proxy
    "intraday_range_pct",     # was: spy_intraday_range
    "spy_return_from_open",
    "spy_vwap_distance",
    "gap_size_pct",
    "breadth_sectors_above_vwap",
    "session_hour",
]

# Tier 4: v5 Alpha — Options flow & microstructure (biggest unlock)
TIER4_ALPHA = [
    "gamma_flip_distance",    # dealer regime proximity (KEY SIGNAL)
    "net_option_premium",     # $ conviction (was: net_premium_flow)
    "sweep_volume",           # institutional urgency
    "sweep_intensity",        # normalized sweep rate
    "gex_level",
    "gex_change",
    "gex_momentum",           # was: gex_velocity
    "put_call_ratio",
    "premium_skew",           # call/put premium ratio
]

# Tier 5: Velocity & interaction (transition detection)
TIER5_VELOCITY = [
    "volatility_expansion_rate",
    "momentum_acceleration",
    "volume_delta_15m",
    "return_acceleration",
    "delta_return_5m",
    "delta_vol_expansion",
    "correlation_spike",
    "liquidity_score",
    "vwap_x_momentum",       # interaction: vwap_distance × return_5m
    "vol_x_breadth",          # interaction: vol × breadth
]

# All features in canonical order
FEATURE_KEYS = TIER1_MACRO + TIER2_CROSS_ASSET + TIER3_INTRADAY + TIER4_ALPHA + TIER5_VELOCITY

# Critical features: if >50% missing, reject the prediction
CRITICAL_FEATURES = set(TIER1_MACRO + TIER4_ALPHA[:5])  # macro + top alpha signals

# ══════════════════════════════════════════════════════════════════
# Schema validation
# ══════════════════════════════════════════════════════════════════

def validate_feature_schema(features: dict) -> dict:
    """
    Validate incoming features against canonical schema.
    Returns: { valid: bool, score: float, missing: list, critical_missing: list }
    """
    all_missing = []
    critical_missing = []
    
    for key in FEATURE_KEYS:
        val = features.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            all_missing.append(key)
            if key in CRITICAL_FEATURES:
                critical_missing.append(key)
    
    total = len(FEATURE_KEYS)
    present = total - len(all_missing)
    score = present / total if total > 0 else 0
    
    # Reject if >50% of critical features missing
    critical_coverage = 1 - (len(critical_missing) / len(CRITICAL_FEATURES)) if CRITICAL_FEATURES else 1
    valid = critical_coverage >= 0.5
    
    if critical_missing:
        log.warning(f"  Missing critical features ({len(critical_missing)}): {critical_missing[:5]}")
    
    return {
        "valid": valid,
        "score": round(score, 4),
        "total_features": total,
        "present": present,
        "missing_count": len(all_missing),
        "critical_missing": critical_missing,
        "critical_coverage": round(critical_coverage, 4),
    }


class TacticalModel24h:
    """
    v2 Tactical 24h Forecast Model.
    
    Architecture:
    - Model A (Opportunity): "Is there enough directional edge to act?"
    - Model B (Direction): "Which way?" (binary: up/down)
    - Ridge companion: Linear baseline for ensemble stability
    - Smart Abstain: Suppresses forecast when edge is weak
    """

    def __init__(self):
        self.lgbm_classifier = None      # LGBMClassifier for direction (binary)
        self.lgbm_regressor = None       # LGBMRegressor for magnitude
        self.ridge_classifier = None     # Ridge companion for ensemble
        self.feature_names = []
        self.training_samples = 0
        self.last_trained_at = None
        self.cv_direction_accuracy = 0.0
        self.cv_magnitude_corr = 0.0
        self.cv_ridge_accuracy = 0.0
        
        # Abstain thresholds (self-tuning)
        self.opportunity_threshold = 0.55   # minimum confidence to act
        self.regime_thresholds = {
            "trend": 0.50,         # lower bar in clear trends
            "mean_reversion": 0.55,
            "chop": 0.65,          # higher bar in choppy conditions
            "liquidity_vacuum": 0.70,
        }

    @property
    def is_trained(self):
        return self.lgbm_classifier is not None and self.lgbm_regressor is not None

    def _get_supabase(self):
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    def _fetch_training_data(self, user_id: str, lookback_days: int = 120):
        """Fetch labeled fast_features for training."""
        sb = self._get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        labels_resp = sb.table("fast_feature_labels").select(
            "feature_snapshot_id, spy_return_24h, spy_direction, "
            "abs_move_gt_1pct, vol_expanded, regime_at_label, session_segment"
        ).eq("user_id", user_id).eq("labeled", True).gte("label_at", cutoff).execute()

        if not labels_resp.data:
            return None

        snapshot_ids = [l["feature_snapshot_id"] for l in labels_resp.data]

        features_resp = sb.table("fast_features").select(
            "id, features, spy_price, vix_level"
        ).in_("id", snapshot_ids[:500]).execute()

        if not features_resp.data:
            return None

        feature_map = {f["id"]: f for f in features_resp.data}
        X_rows = []
        y_direction = []  # BINARY: 0=down, 1=up
        y_magnitude = []
        regime_labels = []

        for label in labels_resp.data:
            feat = feature_map.get(label["feature_snapshot_id"])
            if not feat or not feat.get("features"):
                continue

            features = feat["features"]
            
            # Schema validation per sample
            row = []
            for k in FEATURE_KEYS:
                val = features.get(k)
                if val is None or (isinstance(val, (int, float)) and not np.isfinite(float(val))):
                    row.append(0.0)
                elif isinstance(val, str):
                    # Encode categorical (intraday_regime, session_segment)
                    row.append(0.0)  # skip string features for now
                else:
                    row.append(float(val))

            # Add spy_price and vix_level as bonus features
            row.append(feat.get("spy_price") or 0)
            row.append(feat.get("vix_level") or 0)

            X_rows.append(row)

            # BINARY direction: skip "flat" samples OR map flat → abstain training
            direction = label.get("spy_direction", "flat")
            spy_return = label.get("spy_return_24h", 0) or 0
            
            # Binary: use actual return sign (more reliable than string label)
            if abs(spy_return) < 0.05:  # <0.05% = effectively flat → skip for direction training
                y_direction.append(-1)  # mark for removal
            else:
                y_direction.append(1 if spy_return > 0 else 0)
            
            y_magnitude.append(abs(spy_return))
            regime_labels.append(label.get("regime_at_label", "neutral"))

        if len(X_rows) < MIN_TACTICAL_SAMPLES:
            return None

        feature_names = FEATURE_KEYS + ["spy_price", "vix_level"]
        X = np.array(X_rows, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_dir = np.array(y_direction)
        y_mag = np.array(y_magnitude, dtype=np.float64)

        return X, y_dir, y_mag, feature_names, regime_labels

    def train(self, user_id: str, lookback_days: int = 120) -> dict:
        """Train direction classifier + magnitude regressor + ridge companion."""
        result = self._fetch_training_data(user_id, lookback_days)

        if result is None:
            return {
                "status": "insufficient_data",
                "min_required": MIN_TACTICAL_SAMPLES,
                "model_version": MODEL_VERSION,
            }

        X, y_dir, y_mag, feature_names, regime_labels = result
        self.feature_names = feature_names

        # Filter out "flat" samples for direction training
        dir_mask = y_dir >= 0
        X_dir = X[dir_mask]
        y_dir_clean = y_dir[dir_mask]

        log.info(f"Tactical-24h training: {X.shape[0]} total, {X_dir.shape[0]} directional, {X.shape[1]} features")

        if len(X_dir) < 100:
            return {"status": "insufficient_directional_data", "total": len(X), "directional": len(X_dir)}

        # ── Walk-Forward Split (80/20) ──
        # TODO: Replace with PurgedWalkForwardCV (purge=2d, embargo=1d)
        split_idx = int(len(X_dir) * 0.8)
        X_train, X_val = X_dir[:split_idx], X_dir[split_idx:]
        y_dir_train, y_dir_val = y_dir_clean[:split_idx], y_dir_clean[split_idx:]

        # Magnitude uses ALL samples (including flat)
        mag_split = int(len(X) * 0.8)
        X_mag_train, X_mag_val = X[:mag_split], X[mag_split:]
        y_mag_train, y_mag_val = y_mag[:mag_split], y_mag[mag_split:]

        if len(X_train) < 50 or len(X_val) < 20:
            return {"status": "insufficient_data", "reason": "not_enough_for_split"}

        # ── 1. LightGBM Direction Classifier (BINARY) ──
        clf = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.025, max_depth=5,
            num_leaves=20, min_child_samples=max(3, len(X_train) // 25),
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.15, reg_lambda=0.15,
            objective="binary",
            verbose=-1,
        )
        clf.fit(X_train, y_dir_train)
        dir_preds = clf.predict(X_val)
        lgbm_accuracy = float(accuracy_score(y_dir_val, dir_preds))

        # ── 2. Ridge Companion Classifier ──
        ridge = RidgeClassifier(alpha=1.0)
        ridge.fit(X_train, y_dir_train)
        ridge_preds = ridge.predict(X_val)
        ridge_accuracy = float(accuracy_score(y_dir_val, ridge_preds))

        # ── 3. LightGBM Magnitude Regressor ──
        reg = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(X_mag_train) // 20),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
        )
        reg.fit(X_mag_train, y_mag_train)
        mag_preds = reg.predict(X_mag_val)
        mag_corr = float(np.corrcoef(mag_preds, y_mag_val)[0, 1])
        if np.isnan(mag_corr):
            mag_corr = 0.0

        # Store models
        self.lgbm_classifier = clf
        self.lgbm_regressor = reg
        self.ridge_classifier = ridge
        self.training_samples = len(X)
        self.last_trained_at = datetime.utcnow().isoformat()
        self.cv_direction_accuracy = lgbm_accuracy
        self.cv_magnitude_corr = mag_corr
        self.cv_ridge_accuracy = ridge_accuracy

        # Feature importance
        clf_importance = dict(zip(feature_names, [int(x) for x in clf.feature_importances_]))
        reg_importance = dict(zip(feature_names, [int(x) for x in reg.feature_importances_]))

        # Identify top alpha contributors
        sorted_imp = sorted(clf_importance.items(), key=lambda x: x[1], reverse=True)
        alpha_feature_contribution = {k: v for k, v in sorted_imp if k in [f for f in TIER4_ALPHA]}

        log.info(
            f"Tactical-24h trained: lgbm_acc={lgbm_accuracy:.3f}, ridge_acc={ridge_accuracy:.3f}, "
            f"mag_corr={mag_corr:.3f}, samples={len(X)}, directional={len(X_dir)}"
        )

        return {
            "status": "trained",
            "model_version": MODEL_VERSION,
            "training_samples": len(X),
            "directional_samples": len(X_dir),
            "flat_samples_excluded": len(X) - len(X_dir),
            "lgbm_direction_accuracy": round(lgbm_accuracy, 4),
            "ridge_direction_accuracy": round(ridge_accuracy, 4),
            "ensemble_agreement": round(float(np.mean(dir_preds == ridge_preds)), 4),
            "magnitude_correlation": round(mag_corr, 4),
            "classifier_importance": clf_importance,
            "regressor_importance": reg_importance,
            "alpha_feature_contribution": alpha_feature_contribution,
            "feature_count": len(feature_names),
            "feature_tiers": {
                "tier1_macro": len(TIER1_MACRO),
                "tier2_cross_asset": len(TIER2_CROSS_ASSET),
                "tier3_intraday": len(TIER3_INTRADAY),
                "tier4_alpha": len(TIER4_ALPHA),
                "tier5_velocity": len(TIER5_VELOCITY),
            },
            "trained_at": self.last_trained_at,
        }

    def predict(self, features: dict) -> dict:
        """
        Predict 24h direction + magnitude with decision layer.
        
        Decision flow:
        1. Schema validation → reject if critical features missing
        2. LightGBM + Ridge predictions → ensemble direction
        3. Disagreement filter → abstain if models disagree AND confidence low
        4. Opportunity filter → abstain if confidence below regime threshold
        5. Magnitude estimate
        6. Calibrated confidence output
        """
        if not self.is_trained:
            return {"status": "not_trained", "model_version": MODEL_VERSION}

        # ── 1. Schema Validation ──
        schema_check = validate_feature_schema(features)
        if not schema_check["valid"]:
            return {
                "status": "rejected",
                "reason": "critical_features_missing",
                "schema_check": schema_check,
                "model_version": MODEL_VERSION,
            }

        # ── Build feature vector ──
        row = []
        for k in FEATURE_KEYS:
            val = features.get(k)
            if val is None or (isinstance(val, (int, float)) and not np.isfinite(float(val))):
                row.append(0.0)
            elif isinstance(val, str):
                row.append(0.0)
            else:
                row.append(float(val))
        
        row.append(features.get("spy_price", 0) or 0)
        row.append(features.get("vix_level", 0) or 0)

        X = np.array([row], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── 2. LightGBM Prediction ──
        lgbm_proba = self.lgbm_classifier.predict_proba(X)[0]  # [down_prob, up_prob]
        lgbm_direction = "up" if lgbm_proba[1] > 0.5 else "down"
        lgbm_confidence = float(max(lgbm_proba))

        # ── 3. Ridge Prediction ──
        ridge_pred = int(self.ridge_classifier.predict(X)[0])
        ridge_direction = "up" if ridge_pred == 1 else "down"
        # Ridge doesn't have predict_proba, use decision function as proxy
        ridge_score = float(self.ridge_classifier.decision_function(X)[0])
        ridge_confidence = min(1.0, abs(ridge_score) / 3.0)  # normalize

        # ── 4. Ensemble + Disagreement Filter ──
        models_agree = lgbm_direction == ridge_direction
        
        # Weighted ensemble: LightGBM 70%, Ridge 30%
        if models_agree:
            direction = lgbm_direction
            ensemble_confidence = 0.7 * lgbm_confidence + 0.3 * ridge_confidence
        else:
            # Disagreement: use LightGBM but penalize confidence
            direction = lgbm_direction
            ensemble_confidence = lgbm_confidence * 0.6  # heavy penalty

        # ── 5. Regime-Aware Opportunity Filter ──
        intraday_regime = features.get("intraday_regime", "chop")
        if isinstance(intraday_regime, str):
            threshold = self.regime_thresholds.get(intraday_regime, self.opportunity_threshold)
        else:
            threshold = self.opportunity_threshold

        should_abstain = False
        abstain_reason = None

        if ensemble_confidence < threshold:
            should_abstain = True
            abstain_reason = f"low_confidence ({ensemble_confidence:.3f} < {threshold:.3f})"
        elif not models_agree and ensemble_confidence < 0.6:
            should_abstain = True
            abstain_reason = f"disagreement + low_confidence"

        # ── 6. Magnitude Prediction ──
        magnitude = float(self.lgbm_regressor.predict(X)[0])
        magnitude = max(0, min(10, magnitude))

        # ── 7. Tail Probabilities ──
        up_prob = float(lgbm_proba[1]) if len(lgbm_proba) > 1 else 0.5
        down_prob = float(lgbm_proba[0]) if len(lgbm_proba) > 0 else 0.5
        tail_up = up_prob * min(1, magnitude / 2.0)
        tail_down = down_prob * min(1, magnitude / 2.0)

        return {
            "status": "ok",
            "direction": "abstain" if should_abstain else direction,
            "raw_direction": direction,  # always present even during abstain
            "direction_confidence": round(ensemble_confidence, 4),
            "direction_probabilities": {
                "down": round(down_prob, 4),
                "up": round(up_prob, 4),
            },
            "magnitude_estimate": round(magnitude, 4),
            "tail_probabilities": {
                "up_2pct": round(tail_up, 4),
                "down_2pct": round(tail_down, 4),
            },
            "decision_layer": {
                "abstain": should_abstain,
                "abstain_reason": abstain_reason,
                "models_agree": models_agree,
                "lgbm_direction": lgbm_direction,
                "lgbm_confidence": round(lgbm_confidence, 4),
                "ridge_direction": ridge_direction,
                "ridge_confidence": round(ridge_confidence, 4),
                "intraday_regime": intraday_regime,
                "opportunity_threshold": threshold,
            },
            "schema_check": schema_check,
            "model_version": MODEL_VERSION,
            "features_used": len(self.feature_names),
            "training_samples": self.training_samples,
            "cv_direction_accuracy": self.cv_direction_accuracy,
            "cv_ridge_accuracy": self.cv_ridge_accuracy,
            "cv_magnitude_correlation": self.cv_magnitude_corr,
        }


# Singleton
_tactical_model_24h = None

def get_tactical_model_24h() -> TacticalModel24h:
    global _tactical_model_24h
    if _tactical_model_24h is None:
        _tactical_model_24h = TacticalModel24h()
    return _tactical_model_24h
