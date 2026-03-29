"""
Tactical 2h Forecast Model — LightGBM-based classifier + regressor.
Predicts 2-hour market direction, magnitude, continuation/reversal probabilities
using v2 intraday features from fast_features table.

COMPLETELY INDEPENDENT from the 24h tactical model (tactical_model.py).
Separate singleton, separate training data, separate hyperparameters.
"""
import os
import logging
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from supabase import create_client
from datetime import datetime, timedelta

log = logging.getLogger("tactical-model-2h")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MIN_TACTICAL_2H_SAMPLES = 150  # lower than 24h (200) since intraday features are richer
MODEL_VERSION = "tactical-2h-v1"

# ─── Feature keys ────────────────────────────────────────────────────────
# Base features (same as 24h model — shared from fast_features.features JSONB)
BASE_FEATURE_KEYS = [
    "vix_raw", "vix_zscore", "vix_percentile", "vix_acceleration",
    "spy_return_1d", "spy_return_5d", "spy_return_20d",
    "credit_spread_raw", "credit_spread_change", "credit_spread_zscore",
    "dgs2_level", "dgs2_change",
    "yield_curve_slope",
    "days_to_fomc", "days_to_opex", "earnings_density",
    "gex_level", "gex_change", "gex_velocity",
    "put_call_ratio", "put_call_change",
    "iwm_spy_spread", "qqq_spy_spread",
    "realized_vol_proxy",
    "hyg_tlt_spread_change",
    "uup_return", "gld_return",
    "sector_dispersion",
    "dxy_momentum", "tlt_return",
    "spy_intraday_range",
    "net_premium_flow", "iv_rank_spy",
]

# NEW v2 intraday features (Phase 1) — these are what differentiate 2h from 24h
INTRADAY_FEATURE_KEYS = [
    "spy_return_from_open",     # distance from session open
    "spy_return_5m",            # rolling 5-minute return
    "spy_return_15m",           # rolling 15-minute return
    "spy_return_30m",           # rolling 30-minute return
    "spy_return_60m",           # rolling 60-minute return
    "spy_vwap_distance",        # proxy VWAP distance
    "intraday_range_pct",       # today's high-low range
    "session_hour",             # 0-6.5 (hours since 9:30 ET)
    "is_power_hour",            # boolean (after 3:00 PM ET)
    "is_lunch_fade",            # boolean (11:30-1:30 ET)
    "tick_volume_ratio",        # recent vs session average tick count
    "breadth_sectors_above_vwap", # sector ETFs with positive intraday return
    "gap_size_pct",             # open vs previous close
]

# Combined feature set for the 2h model
FEATURE_KEYS_2H = BASE_FEATURE_KEYS + INTRADAY_FEATURE_KEYS


class TacticalModel2h:
    """
    Manages training and prediction for tactical 2h forecasts.
    
    Key differences from 24h TacticalModel:
    - Uses FEATURE_KEYS_2H (base + intraday features)
    - Trains on labeled_2h = true labels (spy_return_2h, spy_direction_2h)
    - Shallower trees, faster learning rate (tuned for short horizon noise)
    - Lower min samples (150 vs 200)
    - Outputs continuation_probability and reversal_probability
    """

    def __init__(self):
        self.classifier = None       # LGBMClassifier for direction
        self.regressor = None        # LGBMRegressor for magnitude
        self.feature_names = []
        self.training_samples = 0
        self.last_trained_at = None
        self.cv_direction_accuracy = 0.0
        self.cv_magnitude_corr = 0.0

    @property
    def is_trained(self):
        return self.classifier is not None and self.regressor is not None

    def _get_supabase(self):
        return create_client(SUPABASE_URL, SUPABASE_KEY)

    def _fetch_training_data(self, user_id: str, lookback_days: int = 60):
        """
        Fetch labeled 2h fast_features for training.
        
        Key difference from 24h: filters on labeled_2h = true,
        uses spy_return_2h and spy_direction_2h for labels.
        Shorter default lookback (60d vs 90d) since intraday data
        accumulates faster (~26 samples/day vs ~12 for 24h).
        """
        sb = self._get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        # Get 2h-labeled snapshots
        labels_resp = sb.table("fast_feature_labels").select(
            "feature_snapshot_id, spy_return_2h, spy_direction_2h, "
            "spy_max_favorable_2h, spy_max_adverse_2h"
        ).eq("user_id", user_id).eq("labeled_2h", True).gte("label_at", cutoff).execute()

        if not labels_resp.data:
            return None, None, None, None

        snapshot_ids = [l["feature_snapshot_id"] for l in labels_resp.data]

        # Batch fetch features (chunk if >500)
        all_features = []
        for i in range(0, len(snapshot_ids), 500):
            chunk = snapshot_ids[i:i+500]
            resp = sb.table("fast_features").select(
                "id, features, spy_price, vix_level"
            ).in_("id", chunk).execute()
            if resp.data:
                all_features.extend(resp.data)

        if not all_features:
            return None, None, None, None

        # Join features to labels
        feature_map = {f["id"]: f for f in all_features}
        X_rows = []
        y_direction = []
        y_magnitude = []

        for label in labels_resp.data:
            feat = feature_map.get(label["feature_snapshot_id"])
            if not feat or not feat.get("features"):
                continue

            features = feat["features"]
            
            # Build feature vector using 2h feature keys (base + intraday)
            row = [features.get(k, 0) or 0 for k in FEATURE_KEYS_2H]

            # Add spy_price and vix_level as additional features
            row.append(feat.get("spy_price") or 0)
            row.append(feat.get("vix_level") or 0)

            X_rows.append(row)

            # Direction: 0=down, 1=flat/sideways, 2=up
            direction = label.get("spy_direction_2h", "sideways")
            y_direction.append({"down": 0, "sideways": 1, "flat": 1, "up": 2}.get(direction, 1))
            
            # Magnitude: absolute 2h return
            y_magnitude.append(abs(label.get("spy_return_2h", 0) or 0))

        if len(X_rows) < MIN_TACTICAL_2H_SAMPLES:
            log.info(f"2h model: only {len(X_rows)} samples, need {MIN_TACTICAL_2H_SAMPLES}")
            return None, None, None, None

        feature_names = FEATURE_KEYS_2H + ["spy_price", "vix_level"]
        X = np.array(X_rows, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, np.array(y_direction), np.array(y_magnitude, dtype=np.float64), feature_names

    def train(self, user_id: str, lookback_days: int = 60) -> dict:
        """
        Train both classifier and regressor on 2h labeled data.
        
        Hyperparameter differences from 24h model:
        - max_depth=3 (shallower — less overfitting on noisy intraday data)
        - learning_rate=0.05 (faster — shorter horizon needs quicker adaptation)
        - n_estimators=150 (fewer — prevent overfitting on noise)
        - min_child_samples scaled to dataset size
        """
        result = self._fetch_training_data(user_id, lookback_days)

        if result is None or result[0] is None:
            return {
                "status": "insufficient_data",
                "min_required": MIN_TACTICAL_2H_SAMPLES,
                "model_version": MODEL_VERSION,
            }

        X, y_dir, y_mag, feature_names = result
        self.feature_names = feature_names

        log.info(f"2h Tactical training: {X.shape[0]} samples, {X.shape[1]} features")

        # Walk-forward split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_dir_train, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
        y_mag_train, y_mag_val = y_mag[:split_idx], y_mag[split_idx:]

        if len(X_train) < 50 or len(X_val) < 20:
            return {"status": "insufficient_data", "reason": "not_enough_for_split"}

        # ─── Direction classifier (tuned for 2h horizon) ───
        clf = lgb.LGBMClassifier(
            n_estimators=150,         # fewer than 24h (200) — less overfitting
            learning_rate=0.05,       # faster than 24h (0.03) — quicker adaptation
            max_depth=3,              # shallower than 24h (4) — less noise capture
            num_leaves=8,             # fewer than 24h (15) — simpler trees
            min_child_samples=max(5, len(X_train) // 15),  # more conservative
            subsample=0.7,            # lower than 24h (0.8) — more regularization
            colsample_bytree=0.7,     # lower — force feature diversity
            reg_alpha=0.2,            # higher L1 — more feature selection
            reg_lambda=0.2,           # higher L2 — smoother predictions
            num_class=3,
            objective="multiclass",
            verbose=-1,
        )
        clf.fit(X_train, y_dir_train)

        dir_preds = clf.predict(X_val)
        dir_accuracy = float(accuracy_score(y_dir_val, dir_preds))

        # ─── Magnitude regressor ───
        reg = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            min_child_samples=max(5, len(X_train) // 15),
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            verbose=-1,
        )
        reg.fit(X_train, y_mag_train)

        mag_preds = reg.predict(X_val)
        mag_corr = float(np.corrcoef(mag_preds, y_mag_val)[0, 1])
        if np.isnan(mag_corr):
            mag_corr = 0.0

        # Store models
        self.classifier = clf
        self.regressor = reg
        self.training_samples = len(X)
        self.last_trained_at = datetime.utcnow().isoformat()
        self.cv_direction_accuracy = dir_accuracy
        self.cv_magnitude_corr = mag_corr

        # Feature importance
        clf_importance = dict(zip(feature_names, [int(x) for x in clf.feature_importances_]))
        reg_importance = dict(zip(feature_names, [int(x) for x in reg.feature_importances_]))

        # Identify top intraday features for diagnostics
        intraday_importance = {
            k: clf_importance.get(k, 0)
            for k in INTRADAY_FEATURE_KEYS
            if k in clf_importance
        }

        log.info(
            f"2h Tactical trained: dir_acc={dir_accuracy:.3f}, mag_corr={mag_corr:.3f}, "
            f"samples={len(X)}, top_intraday={sorted(intraday_importance.items(), key=lambda x: x[1], reverse=True)[:5]}"
        )

        return {
            "status": "trained",
            "model_version": MODEL_VERSION,
            "training_samples": len(X),
            "direction_accuracy": round(dir_accuracy, 4),
            "magnitude_correlation": round(mag_corr, 4),
            "classifier_importance": clf_importance,
            "regressor_importance": reg_importance,
            "intraday_feature_importance": intraday_importance,
            "feature_count": len(feature_names),
            "trained_at": self.last_trained_at,
        }

    def predict(self, features: dict) -> dict:
        """
        Predict 2h direction + magnitude from a feature dict.
        
        Additional outputs vs 24h model:
        - continuation_probability: prob of continuing current trend
        - reversal_probability: prob of reversing current trend
        - These are derived from direction probs + session context
        """
        if not self.is_trained:
            return {
                "status": "not_trained",
                "model_version": MODEL_VERSION,
            }

        # Build feature vector using 2h feature keys
        row = [features.get(k, 0) or 0 for k in FEATURE_KEYS_2H]
        row.append(features.get("spy_price", 0) or 0)
        row.append(features.get("vix_level", 0) or 0)

        X = np.array([row], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Direction prediction with probabilities
        dir_probs = self.classifier.predict_proba(X)[0]  # [down, flat, up]
        dir_pred = int(np.argmax(dir_probs))
        direction_map = {0: "down", 1: "flat", 2: "up"}
        direction = direction_map[dir_pred]
        direction_confidence = float(dir_probs[dir_pred])

        # ─── Abstain mode ───
        # If no direction has >45% confidence, emit "abstain"
        # This prevents low-confidence 2h signals from degrading decisions
        if direction_confidence < 0.45:
            direction = "abstain"

        # Magnitude prediction
        magnitude = float(self.regressor.predict(X)[0])
        magnitude = max(0, min(5, magnitude))  # clamp 0-5% (tighter than 24h's 0-10%)

        # Tail probabilities
        tail_up = float(dir_probs[2] * min(1, magnitude / 1.5))   # scaled for 2h
        tail_down = float(dir_probs[0] * min(1, magnitude / 1.5))

        # ─── Continuation / Reversal probabilities ───
        # Derived from the intraday momentum context
        spy_return_from_open = features.get("spy_return_from_open", 0) or 0
        
        if spy_return_from_open > 0:
            # Currently trending up
            continuation_probability = float(dir_probs[2])  # prob of continuing up
            reversal_probability = float(dir_probs[0])       # prob of reversing to down
        elif spy_return_from_open < 0:
            # Currently trending down
            continuation_probability = float(dir_probs[0])  # prob of continuing down
            reversal_probability = float(dir_probs[2])       # prob of reversing to up
        else:
            # Flat from open
            continuation_probability = float(dir_probs[1])  # prob of staying flat
            reversal_probability = float(max(dir_probs[0], dir_probs[2]))

        return {
            "status": "ok",
            "direction": direction,
            "direction_confidence": round(direction_confidence, 4),
            "direction_probabilities": {
                "down": round(float(dir_probs[0]), 4),
                "flat": round(float(dir_probs[1]), 4),
                "up": round(float(dir_probs[2]), 4),
            },
            "magnitude_estimate": round(magnitude, 4),
            "tail_probabilities": {
                "up_2pct": round(tail_up, 4),
                "down_2pct": round(tail_down, 4),
            },
            "continuation_probability": round(continuation_probability, 4),
            "reversal_probability": round(reversal_probability, 4),
            "model_version": MODEL_VERSION,
            "features_used": len(self.feature_names),
            "training_samples": self.training_samples,
            "cv_direction_accuracy": self.cv_direction_accuracy,
            "cv_magnitude_correlation": self.cv_magnitude_corr,
        }


# ─── Singleton (SEPARATE from 24h model singleton) ───
_tactical_model_2h = None

def get_tactical_model_2h() -> TacticalModel2h:
    global _tactical_model_2h
    if _tactical_model_2h is None:
        _tactical_model_2h = TacticalModel2h()
    return _tactical_model_2h
