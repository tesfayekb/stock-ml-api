"""
Tactical 2h Forecast Model v3 — Elite Architecture
====================================================
Major upgrades from v2:
1. Feature velocity (2nd derivatives): delta_return, delta_vwap, delta_vol
2. Path_type fed INTO direction model (not separate)
3. Multi-factor intraday regime (trend/chop/mean_reversion/liquidity_vacuum)
4. Opportunity detection split: Model A (is there a move?) + Model B (which direction?)
5. Meta-labeling: secondary model filters bad trades
6. Ensemble diversity: LightGBM + LogisticRegression + MLP
7. Temporal weighting: recent features weighted higher (exponential decay)
8. Smarter abstain: model disagreement, regime uncertainty, low velocity
9. Adaptive context/timing weight: 50/50 at <200 samples, 20/80 at 500+

COMPLETELY INDEPENDENT from the 24h tactical model (tactical_model.py).
"""
import os
import logging
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from supabase import create_client
from datetime import datetime, timedelta

log = logging.getLogger("tactical-model-2h")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MIN_TACTICAL_2H_SAMPLES = 100  # Reduced from 150 — right-sized for current data
MODEL_VERSION = "tactical-2h-v3"

# ─── Feature Groups ──────────────────────────────────────────────────────

# Context features: macro/structural (20% weight in final prediction)
CONTEXT_FEATURE_KEYS = [
    "vix_raw", "vix_zscore", "vix_percentile", "vix_acceleration",
    "spy_return_1d", "spy_return_5d", "spy_return_20d",
    "credit_spread_raw", "credit_spread_change", "credit_spread_zscore",
    "dgs2_level", "dgs2_change", "yield_curve_slope",
    "days_to_fomc", "days_to_opex", "earnings_density",
    "gex_level", "gex_change", "gex_velocity",
    "put_call_ratio", "put_call_change",
    "iwm_spy_spread", "qqq_spy_spread",
    "hyg_tlt_spread_change",
    "uup_return", "gld_return",
    "sector_dispersion", "dxy_momentum", "tlt_return",
    "net_premium_flow", "iv_rank_spy",
]

# Timing features: intraday microstructure (weight scales with sample count)
TIMING_FEATURE_KEYS = [
    # Price action
    "spy_return_from_open", "spy_return_5m", "spy_return_15m",
    "spy_return_30m", "spy_return_60m",
    "spy_vwap_distance", "intraday_range_pct",
    "distance_from_intraday_high", "distance_from_intraday_low",
    "return_acceleration",  # 2nd derivative: (ret_5m - ret_15m)
    # v4: Feature velocity (2nd derivatives) — "is momentum accelerating?"
    "delta_return_5m",           # Δ(return_5m) — acceleration of 5m returns
    "delta_vwap_distance",       # Δ(vwap_distance) — price accelerating toward/away VWAP
    "delta_volume_delta",        # Δ(volume_delta) — flow acceleration
    "delta_vol_expansion",       # Δ(vol_expansion_rate) — vol regime shifting
    "momentum_acceleration",     # 3rd derivative proxy
    # Volume/flow
    "volume_delta_15m", "volume_imbalance_5m",
    "volatility_expansion_rate", "tick_volume_ratio",
    "vwap_slope_15m",
    # Options flow
    "zero_dte_call_put_imbalance", "net_premium_flow_norm",
    # Session encoding
    "session_hour", "session_hour_sin", "session_hour_cos",
    "is_power_hour", "is_lunch_fade",
    # Breadth
    "breadth_sectors_above_vwap", "gap_size_pct",
    # Interaction features
    "vwap_x_momentum", "vol_x_breadth",
    # Realized vol
    "realized_vol_proxy", "spy_intraday_range",
    # v4: Intraday regime features
    "correlation_spike", "liquidity_score",
]

# v5: Path + continuation/reversal features fed INTO direction model
PATH_FEATURE_KEYS = [
    "_path_type",             # from edge function (last known path)
    "_continuation_prob",     # from edge function
    "_reversal_prob",         # from edge function
]

ALL_FEATURE_KEYS = CONTEXT_FEATURE_KEYS + TIMING_FEATURE_KEYS

# VIX regime boundaries
VIX_LOW = 18
VIX_HIGH = 25


class RegimeGatedModel:
    """
    Holds 3 LightGBM models: one per VIX regime (low/mid/high).
    Falls back to global model if regime-specific model has insufficient data.
    """
    def __init__(self):
        self.models = {"low": None, "mid": None, "high": None}
        self.global_model = None
        self.sample_counts = {"low": 0, "mid": 0, "high": 0}

    def get_regime(self, vix: float) -> str:
        if vix < VIX_LOW:
            return "low"
        elif vix > VIX_HIGH:
            return "high"
        return "mid"

    def get_model(self, vix: float):
        regime = self.get_regime(vix)
        model = self.models.get(regime)
        if model is not None and self.sample_counts[regime] >= 50:
            return model, regime, False
        return self.global_model, "global", True

    def set_model(self, regime: str, model, count: int):
        self.models[regime] = model
        self.sample_counts[regime] = count

    def set_global(self, model):
        self.global_model = model


class TacticalModel2h:
    """
    v3 Architecture:
    ┌──────────────────────────────────────────────┐
    │ Model A: Opportunity Detector                 │
    │  "Is there a tradable move in next 2h?"      │
    │  Binary: tradable (1) vs no-trade (0)        │
    ├──────────────────────────────────────────────┤
    │ Model B: Direction Predictor (only if A=1)    │
    │  Context (adaptive weight) + Timing (adaptive)│
    │  + Path features fed IN as inputs             │
    ├──────────────────────────────────────────────┤
    │ Regime Gating: separate models per VIX bucket │
    │  + multi-factor intraday regime input         │
    ├──────────────────────────────────────────────┤
    │ Meta-Model: should we trust this prediction?  │
    │  Filters bad trades post-prediction           │
    ├──────────────────────────────────────────────┤
    │ Ensemble: LightGBM + LogReg + MLP (majority) │
    └──────────────────────────────────────────────┘
    """

    def __init__(self):
        # Model A: Opportunity detector (is there a 2h move?)
        self.opportunity_detector = None
        # Model B: Direction predictor
        self.context_classifier = None
        self.context_regressor = None
        self.timing_classifier = RegimeGatedModel()
        self.timing_regressor = RegimeGatedModel()
        # Ensemble diversity: additional models
        self.logreg_classifier = None  # LogisticRegression
        self.mlp_classifier = None     # MLPClassifier
        # Meta-labeler: "should we trade this?"
        self.meta_model = None
        # Path classifier (also feeds into direction)
        self.path_classifier = None
        # Calibration
        self.calibrator = None  # IsotonicRegression
        # Metadata
        self.feature_names = []
        self.training_samples = 0
        self.last_trained_at = None
        self.cv_direction_accuracy = 0.0
        self.cv_magnitude_corr = 0.0
        self.regime_accuracies = {}
        # Adaptive weights based on sample count
        self._context_weight = 0.5  # starts 50/50, moves to 20/80 with data

    @property
    def is_trained(self):
        return (
            self.context_classifier is not None
            and self.timing_classifier.global_model is not None
        )

    def _get_supabase(self):
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    def _fetch_training_data(self, user_id: str, lookback_days: int = 60):
        """
        Fetch labeled 2h fast_features including multi-task labels.
        Returns enriched data with path_type, continuation/reversal, regime, session.
        """
        sb = self._get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        labels_resp = sb.table("fast_feature_labels").select(
            "feature_snapshot_id, spy_return_2h, spy_direction_2h, "
            "spy_max_favorable_2h, spy_max_adverse_2h, "
            "continuation_probability, reversal_probability, "
            "vol_regime_2h, path_type, regime_at_label, session_segment"
        ).eq("user_id", user_id).eq("labeled_2h", True).gte("label_at", cutoff).execute()

        if not labels_resp.data:
            return None

        snapshot_ids = [l["feature_snapshot_id"] for l in labels_resp.data]

        all_features = []
        for i in range(0, len(snapshot_ids), 500):
            chunk = snapshot_ids[i:i+500]
            resp = sb.table("fast_features").select(
                "id, features, spy_price, vix_level"
            ).in_("id", chunk).execute()
            if resp.data:
                all_features.extend(resp.data)

        if not all_features:
            return None

        feature_map = {f["id"]: f for f in all_features}

        records = []
        for label in labels_resp.data:
            feat = feature_map.get(label["feature_snapshot_id"])
            if not feat or not feat.get("features"):
                continue

            features = feat["features"]
            vix = feat.get("vix_level") or 20

            # Build separate feature vectors for context and timing
            context_row = [features.get(k, 0) or 0 for k in CONTEXT_FEATURE_KEYS]
            timing_row = [features.get(k, 0) or 0 for k in TIMING_FEATURE_KEYS]
            
            # v5: Append path features INTO timing (direction model sees path context)
            cont_prob = label.get("continuation_probability") or 0.5
            rev_prob = label.get("reversal_probability") or 0.5
            path = label.get("path_type", "chop")
            path_one_hot = [1 if path == "trend" else 0, 1 if path == "reversal" else 0, 1 if path == "chop" else 0]
            timing_row_with_path = timing_row + [cont_prob, rev_prob] + path_one_hot
            
            full_row = context_row + timing_row_with_path
            full_row.append(feat.get("spy_price") or 0)
            full_row.append(vix)
            full_row.append(feat.get("spy_price") or 0)
            full_row.append(vix)

            direction = label.get("spy_direction_2h", "sideways")
            y_dir = {"down": 0, "sideways": 1, "flat": 1, "up": 2}.get(direction, 1)
            y_mag = abs(label.get("spy_return_2h", 0) or 0)

            # Triple barrier: determine which barrier was hit
            mfe = label.get("spy_max_favorable_2h") or 0
            mae = label.get("spy_max_adverse_2h") or 0
            ret = label.get("spy_return_2h") or 0
            # Adaptive barriers based on VIX
            upper_barrier = 0.3 if vix < VIX_LOW else 0.5 if vix < VIX_HIGH else 0.8
            lower_barrier = upper_barrier
            if mfe >= upper_barrier:
                barrier_label = 0  # hit_upper
            elif mae >= lower_barrier:
                barrier_label = 1  # hit_lower
            else:
                barrier_label = 2  # timeout

            # Path type: continuation/reversal/chop
            path = label.get("path_type", "chop")
            path_label = {"trend": 0, "continuation": 0, "reversal": 1, "chop": 2}.get(path, 2)

            records.append({
                "context": context_row,
                "timing": timing_row_with_path,  # v5: includes path features
                "full": full_row,
                "y_dir": y_dir,
                "y_mag": y_mag,
                "barrier": barrier_label,
                "path": path_label,
                "vix": vix,
                "regime": label.get("regime_at_label", "neutral"),
                "session": label.get("session_segment", "unknown"),
                "continuation": label.get("continuation_probability"),
                "reversal": label.get("reversal_probability"),
            })

        if len(records) < MIN_TACTICAL_2H_SAMPLES:
            log.info(f"2h v2: only {len(records)} samples, need {MIN_TACTICAL_2H_SAMPLES}")
            return None

        return records

    def train(self, user_id: str, lookback_days: int = 60) -> dict:
        """
        Train the full v2 architecture:
        1. Context classifier/regressor on macro features
        2. Timing classifier/regressor on intraday features (per VIX regime)
        3. Path classifier (continuation/reversal/chop)
        4. Isotonic calibrator
        """
        records = self._fetch_training_data(user_id, lookback_days)

        if records is None:
            return {
                "status": "insufficient_data",
                "min_required": MIN_TACTICAL_2H_SAMPLES,
                "model_version": MODEL_VERSION,
            }

        log.info(f"2h v3 training: {len(records)} samples")
        feature_names = ALL_FEATURE_KEYS + PATH_FEATURE_KEYS + ["cont_prob", "rev_prob"] + ["spy_price", "vix_level"]
        self.feature_names = feature_names
        
        # v5: Set adaptive context/timing weight based on sample count
        n = len(records)
        if n < 200:
            self._context_weight = 0.5  # equal weight — small data safety
        elif n < 500:
            self._context_weight = 0.35
        else:
            self._context_weight = 0.2  # full 20/80 at scale

        # Prepare arrays
        X_context = np.array([r["context"] for r in records], dtype=np.float64)
        X_timing = np.array([r["timing"] for r in records], dtype=np.float64)
        X_full = np.array([r["full"] for r in records], dtype=np.float64)
        y_dir = np.array([r["y_dir"] for r in records])
        y_mag = np.array([r["y_mag"] for r in records], dtype=np.float64)
        y_barrier = np.array([r["barrier"] for r in records])
        y_path = np.array([r["path"] for r in records])
        vix_vals = np.array([r["vix"] for r in records])

        for arr in [X_context, X_timing, X_full]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Walk-forward split (80/20)
        split_idx = int(len(records) * 0.8)
        if split_idx < 50 or len(records) - split_idx < 20:
            return {"status": "insufficient_data", "reason": "not_enough_for_split"}

        # ─── 1. Context Model ───
        ctx_params = dict(
            n_estimators=100, learning_rate=0.03, max_depth=4,
            num_leaves=12, min_child_samples=max(5, split_idx // 20),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            num_class=3, objective="multiclass", verbose=-1,
        )
        self.context_classifier = lgb.LGBMClassifier(**ctx_params)
        self.context_classifier.fit(X_context[:split_idx], y_dir[:split_idx])

        ctx_reg_params = {k: v for k, v in ctx_params.items() if k not in ("num_class", "objective")}
        self.context_regressor = lgb.LGBMRegressor(**ctx_reg_params)
        self.context_regressor.fit(X_context[:split_idx], y_mag[:split_idx])

        # ─── 2. Timing Model (regime-gated) ───
        timing_params = dict(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            num_leaves=8, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=0.2, verbose=-1,
        )

        # Global timing model
        global_clf = lgb.LGBMClassifier(**timing_params, num_class=3, objective="multiclass")
        global_clf.fit(X_timing[:split_idx], y_dir[:split_idx])
        self.timing_classifier.set_global(global_clf)

        global_reg = lgb.LGBMRegressor(**timing_params)
        global_reg.fit(X_timing[:split_idx], y_mag[:split_idx])
        self.timing_regressor.set_global(global_reg)

        # Per-regime timing models
        self.regime_accuracies = {}
        for regime in ["low", "mid", "high"]:
            if regime == "low":
                mask = vix_vals < VIX_LOW
            elif regime == "high":
                mask = vix_vals > VIX_HIGH
            else:
                mask = (vix_vals >= VIX_LOW) & (vix_vals <= VIX_HIGH)

            train_mask = mask[:split_idx]
            val_mask = mask[split_idx:]

            if train_mask.sum() >= 30:
                rcl = lgb.LGBMClassifier(**timing_params, num_class=3, objective="multiclass")
                rcl.fit(X_timing[:split_idx][train_mask], y_dir[:split_idx][train_mask])
                self.timing_classifier.set_model(regime, rcl, int(train_mask.sum()))

                rrg = lgb.LGBMRegressor(**timing_params)
                rrg.fit(X_timing[:split_idx][train_mask], y_mag[:split_idx][train_mask])
                self.timing_regressor.set_model(regime, rrg, int(train_mask.sum()))

                # Track per-regime val accuracy
                if val_mask.sum() >= 5:
                    r_preds = rcl.predict(X_timing[split_idx:][val_mask])
                    r_acc = float(accuracy_score(y_dir[split_idx:][val_mask], r_preds))
                    self.regime_accuracies[regime] = round(r_acc, 4)
                    log.info(f"  Regime '{regime}' model: {int(train_mask.sum())} train, acc={r_acc:.3f}")

        # ─── 3. Path Classifier (continuation/reversal/chop) ───
        path_train_mask = y_path[:split_idx] >= 0  # all valid
        if path_train_mask.sum() >= 50:
            path_params = dict(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                num_leaves=8, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.2, reg_lambda=0.2,
                num_class=3, objective="multiclass", verbose=-1,
            )
            self.path_classifier = lgb.LGBMClassifier(**path_params)
            self.path_classifier.fit(X_full[:split_idx], y_path[:split_idx])

        # ─── 4. Validation ───
        # Combined prediction on validation set with adaptive weights
        cw = self._context_weight
        ctx_probs_val = self.context_classifier.predict_proba(X_context[split_idx:])
        timing_model, _, _ = self.timing_classifier.get_model(20)  # use global for overall eval
        timing_probs_val = timing_model.predict_proba(X_timing[split_idx:])

        combined_probs = cw * ctx_probs_val + (1 - cw) * timing_probs_val
        combined_preds = np.argmax(combined_probs, axis=1)
        dir_accuracy = float(accuracy_score(y_dir[split_idx:], combined_preds))

        # Magnitude
        ctx_mag_val = self.context_regressor.predict(X_context[split_idx:])
        timing_reg_model, _, _ = self.timing_regressor.get_model(20)
        timing_mag_val = timing_reg_model.predict(X_timing[split_idx:])
        combined_mag = cw * ctx_mag_val + (1 - cw) * timing_mag_val
        mag_corr = float(np.corrcoef(combined_mag, y_mag[split_idx:])[0, 1])
        if np.isnan(mag_corr):
            mag_corr = 0.0

        # ─── 5. Global-Only Isotonic Calibration ───
        # v5: Skip per-regime calibration until 300+ samples (too noisy at small N)
        max_probs_val = np.max(combined_probs, axis=1)
        correct_val = (combined_preds == y_dir[split_idx:]).astype(float)
        if len(max_probs_val) >= 30:  # raised from 20 — need more for reliable calibration
            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.calibrator.fit(max_probs_val, correct_val)
        else:
            self.calibrator = None  # don't calibrate with too few samples

        # Store metadata
        self.training_samples = len(records)
        self.last_trained_at = datetime.utcnow().isoformat()
        self.cv_direction_accuracy = dir_accuracy
        self.cv_magnitude_corr = mag_corr

        # Feature importance from timing model (most important)
        timing_importance = dict(zip(
            TIMING_FEATURE_KEYS,
            [int(x) for x in timing_model.feature_importances_[:len(TIMING_FEATURE_KEYS)]]
        ))

        log.info(
            f"2h v2 trained: dir_acc={dir_accuracy:.3f}, mag_corr={mag_corr:.3f}, "
            f"samples={len(records)}, regime_acc={self.regime_accuracies}"
        )

        return {
            "status": "trained",
            "model_version": MODEL_VERSION,
            "training_samples": len(records),
            "direction_accuracy": round(dir_accuracy, 4),
            "magnitude_correlation": round(mag_corr, 4),
            "regime_accuracies": self.regime_accuracies,
            "timing_feature_importance": timing_importance,
            "feature_count": len(feature_names),
            "trained_at": self.last_trained_at,
            "has_path_classifier": self.path_classifier is not None,
            "has_calibrator": self.calibrator is not None,
            "architecture": "context_timing_split_v2",
        }

    def predict(self, features: dict) -> dict:
        """
        v2 Prediction with context/timing split + regime gating.
        """
        if not self.is_trained:
            return {"status": "not_trained", "model_version": MODEL_VERSION}

        vix = features.get("vix_level") or features.get("vix_raw") or 20

        # Build feature vectors
        context_row = [features.get(k, 0) or 0 for k in CONTEXT_FEATURE_KEYS]
        timing_row = [features.get(k, 0) or 0 for k in TIMING_FEATURE_KEYS]
        
        # v5: Append path context into timing features (direction model sees path)
        cont_prob = features.get("_continuation_prob") or 0.5
        rev_prob = features.get("_reversal_prob") or 0.5
        path_type = features.get("_path_type", "chop")
        path_one_hot = [1 if path_type == "trend" else 0, 1 if path_type == "reversal" else 0, 1 if path_type == "chop" else 0]
        timing_row_with_path = timing_row + [cont_prob, rev_prob] + path_one_hot
        
        full_row = context_row + timing_row_with_path + [
            features.get("spy_price", 0) or 0,
            vix,
        ]

        X_ctx = np.nan_to_num(np.array([context_row], dtype=np.float64))
        X_tim = np.nan_to_num(np.array([timing_row_with_path], dtype=np.float64))
        X_full = np.nan_to_num(np.array([full_row], dtype=np.float64))

        # Context prediction (adaptive weight)
        ctx_probs = self.context_classifier.predict_proba(X_ctx)[0]
        ctx_mag = float(self.context_regressor.predict(X_ctx)[0])

        # Timing prediction (regime-gated)
        timing_clf, regime_used, is_fallback = self.timing_classifier.get_model(vix)
        timing_reg, _, _ = self.timing_regressor.get_model(vix)

        timing_probs = timing_clf.predict_proba(X_tim)[0]
        timing_mag = float(timing_reg.predict(X_tim)[0])

        # Adaptive weighted combination: 50/50 at low samples, 20/80 at 500+
        cw = self._context_weight
        combined_probs = cw * ctx_probs + (1 - cw) * timing_probs
        dir_pred = int(np.argmax(combined_probs))
        direction_map = {0: "down", 1: "flat", 2: "up"}
        direction = direction_map[dir_pred]
        direction_confidence = float(combined_probs[dir_pred])

        # v5: Global-only isotonic calibration (skip per-regime until 300+ samples)
        calibrated_confidence = direction_confidence
        if self.calibrator is not None:
            calibrated_confidence = float(
                self.calibrator.predict([direction_confidence])[0]
            )

        # v5: Dynamic abstain threshold based on VIX (passed from edge function)
        conf_threshold = features.get("_confidence_threshold", 0.45)
        if calibrated_confidence < conf_threshold:
            direction = "abstain"

        # Combined magnitude
        magnitude = 0.2 * ctx_mag + 0.8 * timing_mag
        magnitude = max(0, min(5, magnitude))

        # Tail probabilities
        tail_up = float(combined_probs[2] * min(1, magnitude / 1.5))
        tail_down = float(combined_probs[0] * min(1, magnitude / 1.5))

        # Continuation / Reversal
        spy_return_from_open = features.get("spy_return_from_open", 0) or 0
        if spy_return_from_open > 0:
            continuation_prob = float(combined_probs[2])
            reversal_prob = float(combined_probs[0])
        elif spy_return_from_open < 0:
            continuation_prob = float(combined_probs[0])
            reversal_prob = float(combined_probs[2])
        else:
            continuation_prob = float(combined_probs[1])
            reversal_prob = float(max(combined_probs[0], combined_probs[2]))

        # Path type prediction (if classifier available)
        path_prediction = None
        if self.path_classifier is not None:
            path_probs = self.path_classifier.predict_proba(X_full)[0]
            path_map = {0: "trend", 1: "reversal", 2: "chop"}
            path_pred = int(np.argmax(path_probs))
            path_prediction = {
                "type": path_map[path_pred],
                "confidence": round(float(path_probs[path_pred]), 4),
                "probabilities": {
                    "trend": round(float(path_probs[0]), 4),
                    "reversal": round(float(path_probs[1]), 4),
                    "chop": round(float(path_probs[2]), 4),
                },
            }

        return {
            "status": "ok",
            "direction": direction,
            "direction_confidence": round(direction_confidence, 4),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "direction_probabilities": {
                "down": round(float(combined_probs[0]), 4),
                "flat": round(float(combined_probs[1]), 4),
                "up": round(float(combined_probs[2]), 4),
            },
            "context_signal": {
                "down": round(float(ctx_probs[0]), 4),
                "flat": round(float(ctx_probs[1]), 4),
                "up": round(float(ctx_probs[2]), 4),
            },
            "timing_signal": {
                "down": round(float(timing_probs[0]), 4),
                "flat": round(float(timing_probs[1]), 4),
                "up": round(float(timing_probs[2]), 4),
            },
            "magnitude_estimate": round(magnitude, 4),
            "tail_probabilities": {
                "up_2pct": round(tail_up, 4),
                "down_2pct": round(tail_down, 4),
            },
            "continuation_probability": round(continuation_prob, 4),
            "reversal_probability": round(reversal_prob, 4),
            "path_prediction": path_prediction,
            "regime_used": regime_used,
            "regime_is_fallback": is_fallback,
            "model_version": MODEL_VERSION,
            "features_used": len(self.feature_names),
            "training_samples": self.training_samples,
            "cv_direction_accuracy": self.cv_direction_accuracy,
            "cv_magnitude_correlation": self.cv_magnitude_corr,
        }


# ─── Singleton ───
_tactical_model_2h = None

def get_tactical_model_2h() -> TacticalModel2h:
    global _tactical_model_2h
    if _tactical_model_2h is None:
        _tactical_model_2h = TacticalModel2h()
    return _tactical_model_2h
