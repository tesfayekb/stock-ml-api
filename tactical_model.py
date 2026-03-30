"""
Tactical 24h Forecast Model v3 — Elite Upgrade
LightGBM + Ridge ensemble with:
  - Regime-conditioned routing
  - Isotonic calibration
  - 102-feature canonical schema (v7 poller aligned)
  - Feature importance pruning support
  - Forecast quality model
  - State transition awareness

Created: 2026-03-29 (v1)
Updated: 2026-03-30 (v3 — Elite Upgrade)
"""
import os
import logging
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss
from supabase import create_client
from datetime import datetime, timedelta

log = logging.getLogger("tactical-24h-v3")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MIN_TACTICAL_SAMPLES = 200  # minimum labeled snapshots for training
MODEL_VERSION = "tactical-24h-v3"

# ═══════════════════════════════════════════════════════════════════════
#  102-Feature Canonical Schema (aligned with fast-feature-poller v7)
# ═══════════════════════════════════════════════════════════════════════

FEATURE_KEYS = [
    # ── Tier 1: Macro Regime (always available) ──
    "vix_raw", "vix_zscore", "vix_percentile", "vix_acceleration",
    "vix_term_structure", "vix_term_consistency",
    "spy_return_1d", "spy_return_5d", "spy_return_20d",
    "credit_spread_raw", "credit_spread_change", "credit_spread_zscore",
    "dgs2_level", "dgs2_change",
    "yield_curve_slope",
    "hyg_tlt_spread_change",
    "uup_return", "gld_return",
    "dxy_momentum", "tlt_return",

    # ── Tier 2: Market Structure ──
    "gex_level", "gex_change", "gex_velocity",
    "put_call_ratio", "put_call_change",
    "iwm_spy_spread", "qqq_spy_spread",
    "realized_vol_proxy",
    "sector_dispersion",
    "spy_intraday_range",

    # ── Tier 3: Calendar / Event Proximity ──
    "days_to_fomc", "days_to_opex", "earnings_density",
    "days_to_cpi", "days_to_nfp",
    "is_fomc_day", "is_opex_day", "is_cpi_day", "is_nfp_day",

    # ── Tier 4: Options Flow (Unusual Whales) ──
    "net_premium_flow", "iv_rank_spy",
    "gamma_flip_distance", "sweep_intensity", "net_option_premium",

    # ── Tier 5: Close-Structure (high alpha for 24h) ──
    "close_vs_vwap", "close_vs_day_high", "close_vs_day_low",
    "late_day_acceleration", "gap_fill_pct",
    "upper_wick_ratio", "lower_wick_ratio", "body_ratio",

    # ── Tier 6: Velocity / Acceleration ──
    "return_acceleration", "volatility_expansion",
    "volume_delta_proxy", "vwap_slope",
    "delta_return_5m", "momentum_divergence",

    # ── Tier 7: Internals / Breadth ──
    "up_down_volume_ratio", "equal_weight_vs_cap_weight",
    "advance_decline_proxy", "new_highs_lows_ratio",
    "breadth_thrust_proxy",

    # ── Tier 8: Intraday Session ──
    "session_hour_sin", "session_hour_cos",
    "session_segment_encoded",
    "overnight_return", "opening_drive",

    # ── Tier 9: Temporal Awareness (v7 additions) ──
    "day_of_week", "day_of_week_sin", "day_of_week_cos",
    "is_monday", "is_friday",
    "is_opex_week", "is_month_end", "is_quarter_end",
    "days_since_last_fomc", "days_until_next_fomc",

    # ── Tier 10: State Transition Awareness (v7 additions) ──
    "regime_change_signal", "volatility_regime_shift",
    "momentum_breakdown_signal", "trend_exhaustion_score",
    "regime_persistence_days", "cross_asset_divergence",
    "yield_curve_velocity", "credit_stress_acceleration",
    "vix_regime_transition_prob",

    # ── Tier 11: Cross-Asset Confirmation ──
    "gold_dollar_divergence", "bond_equity_correlation",
    "em_developed_spread", "commodity_momentum",

    # ── Tier 12: Derived Ratios ──
    "vix_spy_ratio", "credit_vix_ratio",
    "flow_momentum_ratio", "breadth_volatility_ratio",
]


# ═══════════════════════════════════════════════════════════════════════
#  Regime-Specific Thresholds
# ═══════════════════════════════════════════════════════════════════════

REGIME_THRESHOLDS = {
    "trend":    {"opportunity": 0.50, "confidence_min": 0.55, "abstain_penalty": 0.0},
    "chop":     {"opportunity": 0.65, "confidence_min": 0.62, "abstain_penalty": 0.10},
    "risk_off": {"opportunity": 0.60, "confidence_min": 0.58, "abstain_penalty": 0.05},
    "crisis":   {"opportunity": 0.70, "confidence_min": 0.65, "abstain_penalty": 0.15},
    "normal":   {"opportunity": 0.55, "confidence_min": 0.55, "abstain_penalty": 0.0},
    "risk_on":  {"opportunity": 0.50, "confidence_min": 0.52, "abstain_penalty": 0.0},
}


class TacticalModel24h:
    """
    v3 Tactical 24h Model — Elite Upgrade
    - Binary direction (up/down) + smart abstain
    - LightGBM (70%) + Ridge (30%) ensemble
    - Regime-conditioned decision layer
    - Isotonic probability calibration
    - State transition override
    """

    def __init__(self):
        self.lgb_classifier = None
        self.ridge_classifier = None
        self.regressor = None          # LGBMRegressor for magnitude
        self.calibrator = None         # IsotonicRegression for probability calibration
        self.feature_names = []
        self.active_features = []      # after pruning
        self.pruned_features = []      # features removed by importance analysis
        self.training_samples = 0
        self.last_trained_at = None
        self.cv_direction_accuracy = 0.0
        self.cv_ridge_accuracy = 0.0
        self.cv_magnitude_corr = 0.0
        self.calibration_ready = False
        self.feature_importance_map = {}

    @property
    def is_trained(self):
        return self.lgb_classifier is not None and self.ridge_classifier is not None

    def _get_supabase(self):
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    def _fetch_training_data(self, user_id: str, lookback_days: int = 90):
        """Fetch labeled fast_features for training — v3 with enhanced labels."""
        sb = self._get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        # Get labeled snapshots with v2 enriched labels
        labels_resp = sb.table("fast_feature_labels").select(
            "feature_snapshot_id, spy_return_24h, spy_direction, "
            "abs_move_gt_1pct, vol_expanded, "
            "path_type, continuation_probability, reversal_probability, "
            "regime_at_label"
        ).eq("user_id", user_id).eq("labeled", True).gte("label_at", cutoff).execute()

        if not labels_resp.data:
            return None

        # Get the corresponding features
        snapshot_ids = [l["feature_snapshot_id"] for l in labels_resp.data]

        # Batch fetch (chunk if > 500)
        all_features = []
        for i in range(0, len(snapshot_ids), 500):
            chunk = snapshot_ids[i:i+500]
            features_resp = sb.table("fast_features").select(
                "id, features, spy_price, vix_level"
            ).in_("id", chunk).execute()
            if features_resp.data:
                all_features.extend(features_resp.data)

        if not all_features:
            return None

        # Join features to labels
        feature_map = {f["id"]: f for f in all_features}
        X_rows = []
        y_direction = []
        y_magnitude = []
        y_regime = []
        y_path_type = []

        # Use active features if pruning has been done, otherwise all
        feature_keys = self.active_features if self.active_features else FEATURE_KEYS

        for label in labels_resp.data:
            feat = feature_map.get(label["feature_snapshot_id"])
            if not feat or not feat.get("features"):
                continue

            features = feat["features"]
            ret_24h = label.get("spy_return_24h") or 0

            # Binary direction + smart abstain:
            # Exclude low-magnitude moves (< 0.05%) from directional training
            if abs(ret_24h) < 0.05:
                continue

            row = [features.get(k, 0) or 0 for k in feature_keys]
            row.append(feat.get("spy_price") or 0)
            row.append(feat.get("vix_level") or 0)

            X_rows.append(row)

            # Binary: 0=down, 1=up
            y_direction.append(1 if ret_24h > 0 else 0)
            y_magnitude.append(abs(ret_24h))
            y_regime.append(label.get("regime_at_label", "normal"))
            y_path_type.append(label.get("path_type", "unknown"))

        if len(X_rows) < MIN_TACTICAL_SAMPLES:
            return None

        feature_names = list(feature_keys) + ["spy_price", "vix_level"]
        X = np.array(X_rows, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "X": X,
            "y_direction": np.array(y_direction),
            "y_magnitude": np.array(y_magnitude, dtype=np.float64),
            "y_regime": y_regime,
            "y_path_type": y_path_type,
            "feature_names": feature_names,
        }

    def train(self, user_id: str, lookback_days: int = 90) -> dict:
        """Train v3 ensemble: LightGBM + Ridge + magnitude regressor."""
        data = self._fetch_training_data(user_id, lookback_days)

        if data is None:
            return {
                "status": "insufficient_data",
                "min_required": MIN_TACTICAL_SAMPLES,
                "model_version": MODEL_VERSION,
            }

        X = data["X"]
        y_dir = data["y_direction"]
        y_mag = data["y_magnitude"]
        y_regime = data["y_regime"]
        feature_names = data["feature_names"]
        self.feature_names = feature_names

        log.info(f"Tactical-24h v3 training: {X.shape[0]} samples, {X.shape[1]} features")

        # ── Walk-forward split (80/20) with 2-day purge ──
        split_idx = int(len(X) * 0.8)
        purge_buffer = 2
        train_end = max(0, split_idx - purge_buffer)

        X_train, X_val = X[:train_end], X[split_idx:]
        y_dir_train, y_dir_val = y_dir[:train_end], y_dir[split_idx:]
        y_mag_train, y_mag_val = y_mag[:train_end], y_mag[split_idx:]

        if len(X_train) < 50 or len(X_val) < 20:
            return {"status": "insufficient_data", "reason": "not_enough_for_split"}

        # ── 1. Train LightGBM direction classifier (binary) ──
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=5,
            num_leaves=20, min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.15, reg_lambda=0.15,
            objective="binary",
            verbose=-1,
        )
        lgb_clf.fit(X_train, y_dir_train)

        lgb_preds = lgb_clf.predict(X_val)
        lgb_probs = lgb_clf.predict_proba(X_val)[:, 1]  # P(up)
        lgb_accuracy = float(accuracy_score(y_dir_val, lgb_preds))

        # ── 2. Train Ridge classifier (linear baseline) ──
        ridge_clf = RidgeClassifier(alpha=1.0)
        ridge_clf.fit(X_train, y_dir_train)

        ridge_preds = ridge_clf.predict(X_val)
        ridge_accuracy = float(accuracy_score(y_dir_val, ridge_preds))

        # Ridge decision function → pseudo-probability via sigmoid
        ridge_scores = ridge_clf.decision_function(X_val)
        ridge_probs = 1.0 / (1.0 + np.exp(-ridge_scores))

        # ── 3. Train magnitude regressor ──
        reg = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
        )
        reg.fit(X_train, y_mag_train)

        mag_preds = reg.predict(X_val)
        mag_corr = float(np.corrcoef(mag_preds, y_mag_val)[0, 1])
        if np.isnan(mag_corr):
            mag_corr = 0.0

        # ── 4. Ensemble agreement analysis ──
        ensemble_probs = 0.70 * lgb_probs + 0.30 * ridge_probs
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        ensemble_accuracy = float(accuracy_score(y_dir_val, ensemble_preds))

        model_agreement = float(np.mean(lgb_preds == ridge_preds))

        # ── 5. Isotonic calibration (if enough validation data) ──
        calibrator = None
        calibration_ready = False
        if len(X_val) >= 50:
            try:
                calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                calibrator.fit(ensemble_probs, y_dir_val)
                calibration_ready = True
                log.info("Isotonic calibrator fitted successfully")
            except Exception as e:
                log.warning(f"Calibration failed: {e}")
                calibrator = None

        # ── 6. Feature importance (for pruning) ──
        lgb_importance = lgb_clf.feature_importances_
        importance_dict = {}
        for i, fname in enumerate(feature_names):
            importance_dict[fname] = int(lgb_importance[i])

        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Identify bottom 20% for future pruning (don't prune yet, just report)
        total_features = len(sorted_importance)
        prune_threshold = int(total_features * 0.2)
        prune_candidates = [f[0] for f in sorted_importance[-prune_threshold:] if f[1] == 0]

        # ── Store models ──
        self.lgb_classifier = lgb_clf
        self.ridge_classifier = ridge_clf
        self.regressor = reg
        self.calibrator = calibrator
        self.calibration_ready = calibration_ready
        self.training_samples = len(X)
        self.last_trained_at = datetime.utcnow().isoformat()
        self.cv_direction_accuracy = lgb_accuracy
        self.cv_ridge_accuracy = ridge_accuracy
        self.cv_magnitude_corr = mag_corr
        self.feature_importance_map = importance_dict

        log.info(
            f"Tactical-24h v3 trained: "
            f"lgb_acc={lgb_accuracy:.3f}, ridge_acc={ridge_accuracy:.3f}, "
            f"ensemble_acc={ensemble_accuracy:.3f}, "
            f"mag_corr={mag_corr:.3f}, agreement={model_agreement:.3f}, "
            f"calibrated={calibration_ready}, samples={len(X)}"
        )

        # ── Regime-stratified accuracy (if regime labels available) ──
        regime_accuracy = {}
        val_regimes = y_regime[split_idx:]
        for regime in set(val_regimes):
            mask = np.array([r == regime for r in val_regimes])
            if mask.sum() >= 5:
                regime_acc = float(accuracy_score(y_dir_val[mask], ensemble_preds[mask]))
                regime_accuracy[regime] = {
                    "accuracy": round(regime_acc, 4),
                    "samples": int(mask.sum()),
                }

        return {
            "status": "trained",
            "model_version": MODEL_VERSION,
            "training_samples": len(X),
            "lgb_accuracy": round(lgb_accuracy, 4),
            "ridge_accuracy": round(ridge_accuracy, 4),
            "ensemble_accuracy": round(ensemble_accuracy, 4),
            "model_agreement": round(model_agreement, 4),
            "magnitude_correlation": round(mag_corr, 4),
            "calibration_ready": calibration_ready,
            "feature_count": len(feature_names),
            "feature_importance": dict(sorted_importance[:20]),  # top 20
            "prune_candidates": prune_candidates,
            "regime_accuracy": regime_accuracy,
            "trained_at": self.last_trained_at,
        }

    def predict(self, features: dict, regime_context: dict = None) -> dict:
        """
        Predict 24h direction + magnitude with regime-conditioned decision layer.

        Args:
            features: Latest fast_features.features jsonb
            regime_context: Optional dict with keys:
                - current_regime: str (e.g., "trend", "chop", "risk_off")
                - vix_level: float
                - regime_change_signal: float (0-1)
                - regime_confidence: float (0-1)
        """
        if not self.is_trained:
            return {
                "status": "not_trained",
                "model_version": MODEL_VERSION,
            }

        # Use active features if pruning has been done
        feature_keys = self.active_features if self.active_features else FEATURE_KEYS

        # Build feature vector
        row = [features.get(k, 0) or 0 for k in feature_keys]
        row.append(features.get("spy_price", 0) or 0)
        row.append(features.get("vix_level", 0) or 0)

        X = np.array([row], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Regime context ──
        regime = "normal"
        if regime_context:
            regime = regime_context.get("current_regime", "normal")
        thresholds = REGIME_THRESHOLDS.get(regime, REGIME_THRESHOLDS["normal"])

        # ── State transition override ──
        regime_change_signal = 0.0
        if regime_context:
            regime_change_signal = regime_context.get("regime_change_signal", 0.0)

        if regime_change_signal > 0.7:
            return {
                "status": "ok",
                "direction": "abstain",
                "direction_confidence": 0.0,
                "abstain_reason": f"regime_transition_detected (signal={regime_change_signal:.2f})",
                "magnitude_estimate": 0.0,
                "model_version": MODEL_VERSION,
                "decision_layer": {
                    "abstain": True,
                    "abstain_reason": "regime_change_signal > 0.7",
                    "regime_change_signal": round(regime_change_signal, 4),
                    "regime": regime,
                },
            }

        # ── LightGBM prediction ──
        lgb_prob_up = float(self.lgb_classifier.predict_proba(X)[0][1])

        # ── Ridge prediction ──
        ridge_score = float(self.ridge_classifier.decision_function(X)[0])
        ridge_prob_up = 1.0 / (1.0 + np.exp(-ridge_score))

        # ── Ensemble (70/30) ──
        raw_confidence = 0.70 * lgb_prob_up + 0.30 * ridge_prob_up

        # ── Isotonic calibration ──
        calibrated_confidence = raw_confidence
        if self.calibrator is not None and self.calibration_ready:
            try:
                calibrated_confidence = float(
                    self.calibrator.predict([raw_confidence])[0]
                )
            except Exception:
                calibrated_confidence = raw_confidence

        # ── Direction ──
        direction = "up" if calibrated_confidence >= 0.5 else "down"
        confidence = calibrated_confidence if direction == "up" else (1.0 - calibrated_confidence)

        # ── Model agreement penalty ──
        lgb_direction = "up" if lgb_prob_up >= 0.5 else "down"
        ridge_direction = "up" if ridge_prob_up >= 0.5 else "down"
        models_agree = lgb_direction == ridge_direction

        if not models_agree:
            confidence *= 0.85  # 15% penalty for disagreement

        # ── Regime abstain penalty ──
        confidence -= thresholds["abstain_penalty"]

        # ── Opportunity gate (regime-conditioned) ──
        if confidence < thresholds["opportunity"]:
            return {
                "status": "ok",
                "direction": "abstain",
                "direction_confidence": round(confidence, 4),
                "abstain_reason": f"below_opportunity_threshold ({confidence:.3f} < {thresholds['opportunity']})",
                "raw_confidence": round(raw_confidence, 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
                "magnitude_estimate": 0.0,
                "model_version": MODEL_VERSION,
                "decision_layer": {
                    "abstain": True,
                    "regime": regime,
                    "threshold": thresholds["opportunity"],
                    "models_agree": models_agree,
                    "lgb_prob_up": round(lgb_prob_up, 4),
                    "ridge_prob_up": round(ridge_prob_up, 4),
                },
            }

        # ── Magnitude prediction ──
        magnitude = float(self.regressor.predict(X)[0])
        magnitude = max(0, min(10, magnitude))  # clamp 0-10%

        # ── Tail probabilities ──
        tail_up = confidence * min(1.0, magnitude / 2.0) if direction == "up" else (1 - confidence) * min(1.0, magnitude / 2.0)
        tail_down = (1 - confidence) * min(1.0, magnitude / 2.0) if direction == "up" else confidence * min(1.0, magnitude / 2.0)

        # ── Forecast quality score (6 factors) ──
        forecast_quality = self._compute_forecast_quality(
            confidence=confidence,
            models_agree=models_agree,
            magnitude=magnitude,
            regime=regime,
            regime_change_signal=regime_change_signal,
            features=features,
        )

        return {
            "status": "ok",
            "direction": direction,
            "direction_confidence": round(confidence, 4),
            "raw_confidence": round(raw_confidence, 4),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "direction_probabilities": {
                "up": round(calibrated_confidence, 4),
                "down": round(1.0 - calibrated_confidence, 4),
            },
            "magnitude_estimate": round(magnitude, 4),
            "tail_probabilities": {
                "up_2pct": round(tail_up, 4),
                "down_2pct": round(tail_down, 4),
            },
            "model_version": MODEL_VERSION,
            "features_used": len(self.feature_names),
            "training_samples": self.training_samples,
            "cv_direction_accuracy": self.cv_direction_accuracy,
            "cv_ridge_accuracy": self.cv_ridge_accuracy,
            "cv_magnitude_correlation": self.cv_magnitude_corr,
            "calibration_applied": self.calibration_ready,
            "forecast_quality": round(forecast_quality, 4),
            "decision_layer": {
                "abstain": False,
                "regime": regime,
                "threshold": thresholds["opportunity"],
                "models_agree": models_agree,
                "lgb_prob_up": round(lgb_prob_up, 4),
                "ridge_prob_up": round(ridge_prob_up, 4),
                "regime_change_signal": round(regime_change_signal, 4),
            },
        }

    def _compute_forecast_quality(
        self,
        confidence: float,
        models_agree: bool,
        magnitude: float,
        regime: str,
        regime_change_signal: float,
        features: dict,
    ) -> float:
        """
        6-factor forecast quality score (0-1).
        Used by edge function to gate notifications (threshold >= 0.6).
        """
        scores = []

        # 1. Confidence strength (0-1)
        scores.append(min(1.0, confidence / 0.8))

        # 2. Model agreement (binary)
        scores.append(1.0 if models_agree else 0.4)

        # 3. Magnitude significance (prefer > 0.3%)
        scores.append(min(1.0, magnitude / 0.5))

        # 4. Regime stability (low transition = good)
        scores.append(max(0.0, 1.0 - regime_change_signal))

        # 5. Feature completeness (count non-zero features)
        non_zero = sum(1 for k in FEATURE_KEYS[:30] if (features.get(k) or 0) != 0)
        scores.append(min(1.0, non_zero / 20.0))

        # 6. Calibration bonus
        scores.append(1.0 if self.calibration_ready else 0.7)

        return sum(scores) / len(scores)

    def prune_features(self, min_importance: int = 0, max_prune_pct: float = 0.3):
        """
        Prune low-importance features based on last training's importance map.
        Call after training, before next training cycle.
        Returns list of pruned feature names.
        """
        if not self.feature_importance_map:
            return []

        sorted_features = sorted(
            self.feature_importance_map.items(), key=lambda x: x[1]
        )

        max_prune = int(len(sorted_features) * max_prune_pct)
        to_prune = []

        for fname, importance in sorted_features:
            if importance <= min_importance and len(to_prune) < max_prune:
                # Never prune Tier 1 macro features
                if fname not in FEATURE_KEYS[:20]:
                    to_prune.append(fname)

        # Update active features
        self.pruned_features = to_prune
        self.active_features = [f for f in FEATURE_KEYS if f not in to_prune]

        log.info(f"Pruned {len(to_prune)} features: {to_prune}")
        return to_prune


# ═══════════════════════════════════════════════════════════════════════
#  Singleton
# ═══════════════════════════════════════════════════════════════════════

_tactical_model_24h = None

def get_tactical_model_24h() -> TacticalModel24h:
    global _tactical_model_24h
    if _tactical_model_24h is None:
        _tactical_model_24h = TacticalModel24h()
    return _tactical_model_24h
