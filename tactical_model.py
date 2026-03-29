"""
Tactical 24h Forecast Model — LightGBM-based classifier + regressor.
Predicts next-day market direction, magnitude, and tail probabilities
using continuous features from fast_features table.
"""
import os
import logging
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from supabase import create_client
from datetime import datetime, timedelta

log = logging.getLogger("tactical-model")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MIN_TACTICAL_SAMPLES = 200  # minimum labeled snapshots for training
MODEL_VERSION = "tactical-v1"

# Feature keys expected in fast_features.features jsonb
FEATURE_KEYS = [
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


class TacticalModel:
    """Manages training and prediction for tactical 24h forecasts."""

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

    def _fetch_training_data(self, user_id: str, lookback_days: int = 90):
        """Fetch labeled fast_features for training."""
        sb = self._get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

        # Get labeled snapshots with their features
        labels_resp = sb.table("fast_feature_labels").select(
            "feature_snapshot_id, spy_return_24h, spy_direction, "
            "abs_move_gt_1pct, vol_expanded"
        ).eq("user_id", user_id).eq("labeled", True).gte("label_at", cutoff).execute()

        if not labels_resp.data:
            return None, None, None

        # Get the corresponding features
        snapshot_ids = [l["feature_snapshot_id"] for l in labels_resp.data]

        # Batch fetch (Supabase .in_() has limits, chunk if needed)
        features_resp = sb.table("fast_features").select(
            "id, features, spy_price, vix_level"
        ).in_("id", snapshot_ids[:500]).execute()

        if not features_resp.data:
            return None, None, None

        # Join features to labels
        feature_map = {f["id"]: f for f in features_resp.data}
        X_rows = []
        y_direction = []
        y_magnitude = []

        for label in labels_resp.data:
            feat = feature_map.get(label["feature_snapshot_id"])
            if not feat or not feat.get("features"):
                continue

            features = feat["features"]
            row = [features.get(k, 0) or 0 for k in FEATURE_KEYS]

            # Add spy_price and vix_level as additional features
            row.append(feat.get("spy_price") or 0)
            row.append(feat.get("vix_level") or 0)

            X_rows.append(row)

            # Direction: 0=down, 1=flat, 2=up
            direction = label.get("spy_direction", "flat")
            y_direction.append({"down": 0, "flat": 1, "up": 2}.get(direction, 1))
            y_magnitude.append(abs(label.get("spy_return_24h", 0) or 0))

        if len(X_rows) < MIN_TACTICAL_SAMPLES:
            return None, None, None

        feature_names = FEATURE_KEYS + ["spy_price", "vix_level"]
        X = np.array(X_rows, dtype=np.float64)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, np.array(y_direction), np.array(y_magnitude, dtype=np.float64), feature_names

    def train(self, user_id: str, lookback_days: int = 90) -> dict:
        """Train both classifier and regressor on labeled data."""
        result = self._fetch_training_data(user_id, lookback_days)

        if result is None or result[0] is None:
            return {
                "status": "insufficient_data",
                "min_required": MIN_TACTICAL_SAMPLES,
                "model_version": MODEL_VERSION,
            }

        X, y_dir, y_mag, feature_names = result
        self.feature_names = feature_names

        log.info(f"Tactical training: {X.shape[0]} samples, {X.shape[1]} features")

        # Walk-forward split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_dir_train, y_dir_val = y_dir[:split_idx], y_dir[split_idx:]
        y_mag_train, y_mag_val = y_mag[:split_idx], y_mag[split_idx:]

        if len(X_train) < 50 or len(X_val) < 20:
            return {"status": "insufficient_data", "reason": "not_enough_for_split"}

        # Train direction classifier
        clf = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            num_class=3, objective="multiclass",
            verbose=-1,
        )
        clf.fit(X_train, y_dir_train)

        dir_preds = clf.predict(X_val)
        dir_accuracy = float(accuracy_score(y_dir_val, dir_preds))

        # Train magnitude regressor
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

        log.info(f"Tactical trained: dir_acc={dir_accuracy:.3f}, mag_corr={mag_corr:.3f}, samples={len(X)}")

        return {
            "status": "trained",
            "model_version": MODEL_VERSION,
            "training_samples": len(X),
            "direction_accuracy": round(dir_accuracy, 4),
            "magnitude_correlation": round(mag_corr, 4),
            "classifier_importance": clf_importance,
            "regressor_importance": reg_importance,
            "feature_count": len(feature_names),
            "trained_at": self.last_trained_at,
        }

    def predict(self, features: dict) -> dict:
        """Predict 24h direction + magnitude from a feature dict."""
        if not self.is_trained:
            return {
                "status": "not_trained",
                "model_version": MODEL_VERSION,
            }

        # Build feature vector
        row = [features.get(k, 0) or 0 for k in FEATURE_KEYS]
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

        # Magnitude prediction
        magnitude = float(self.regressor.predict(X)[0])
        magnitude = max(0, min(10, magnitude))  # clamp 0-10%

        # Tail probabilities (heuristic from direction probs + magnitude)
        tail_up = float(dir_probs[2] * min(1, magnitude / 2.0))
        tail_down = float(dir_probs[0] * min(1, magnitude / 2.0))

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
            "model_version": MODEL_VERSION,
            "features_used": len(self.feature_names),
            "training_samples": self.training_samples,
            "cv_direction_accuracy": self.cv_direction_accuracy,
            "cv_magnitude_correlation": self.cv_magnitude_corr,
        }


# Singleton
_tactical_model = None

def get_tactical_model() -> TacticalModel:
    global _tactical_model
    if _tactical_model is None:
        _tactical_model = TacticalModel()
    return _tactical_model
