"""
Gaussian HMM Regime Classifier — 4-state regime detection.

States: risk_on, normal, risk_off, crisis
Features (8D): vix_zscore, vix_term_ratio, spy_return_5d, spy_return_20d,
               credit_spread, yield_curve_slope, cross_asset_momentum, dollar_strength

Hardening (2026-03-23):
  - Covariance regularization after fit to prevent singular matrices
  - NaN/Inf scrubbing in train() and predict()
  - try/except around score_samples/predict_proba to return safe fallback
  - Feature dimension validation before predict
"""

import numpy as np
import pickle
import os
import logging
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger("regime-hmm")

MODEL_PATH = os.environ.get("HMM_MODEL_PATH", "/tmp/regime_hmm.pkl")
STATE_LABELS = ["risk_on", "normal", "risk_off", "crisis"]

# Minimum regularization added to covariance diagonals to prevent singular matrices
COV_REGULARIZATION = 1e-4


def _scrub_features(features: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 and clamp extreme values."""
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -50.0, 50.0)
    return features


class RegimeHMM:
    """Wrapper around hmmlearn GaussianHMM with auto state labeling."""

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model: GaussianHMM | None = None
        self.state_map: dict[int, str] = {}
        self._load_if_exists()

    def _load_if_exists(self):
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved["model"]
                    self.state_map = saved["state_map"]
                    log.info(f"Loaded HMM model from {MODEL_PATH}")
        except Exception as e:
            log.warning(f"Failed to load HMM model: {e}")
            self.model = None
            self.state_map = {}

    def _save(self):
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"model": self.model, "state_map": self.state_map}, f)
            log.info(f"Saved HMM model to {MODEL_PATH}")
        except Exception as e:
            log.error(f"Failed to save HMM model: {e}")

    def _regularize_covariances(self):
        """
        Add small diagonal regularization to all covariance matrices.
        Prevents singular matrix errors during score_samples/predict_proba.
        """
        if self.model is None:
            return
        n_features = self.model.means_.shape[1]
        for i in range(self.n_states):
            self.model.covars_[i] += np.eye(n_features) * COV_REGULARIZATION
        log.info(f"Applied covariance regularization (eps={COV_REGULARIZATION})")

    def train(self, features: np.ndarray, n_iter: int = 200) -> dict:
        """
        Train HMM on historical feature matrix.

        Args:
            features: shape (T, 8) — standardized feature matrix
            n_iter: EM iterations

        Returns:
            dict with state_labels, transition_matrix, means, training_samples
        """
        # Scrub input features
        features = _scrub_features(features)

        # Validate: need at least n_states * 3 samples
        min_samples = self.n_states * 3
        if features.shape[0] < min_samples:
            return {
                "error": f"Need at least {min_samples} samples, got {features.shape[0]}",
                "training_samples": features.shape[0],
                "converged": False,
            }

        # Check for degenerate features (zero variance columns)
        col_std = np.std(features, axis=0)
        degenerate_cols = np.where(col_std < 1e-10)[0]
        if len(degenerate_cols) > 0:
            log.warning(f"Degenerate columns (near-zero variance): {degenerate_cols.tolist()}")
            # Add tiny noise to prevent singular covariance
            for col_idx in degenerate_cols:
                features[:, col_idx] += np.random.normal(0, 1e-6, features.shape[0])

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )
        self.model.fit(features)

        # CRITICAL: Regularize covariances after fit to prevent singular matrices
        self._regularize_covariances()

        # Auto-label states by analyzing emission means
        means = self.model.means_
        self.state_map = self._label_states(means)
        self._save()

        return {
            "state_labels": self.state_map,
            "transition_matrix": self.model.transmat_.tolist(),
            "means": {
                self.state_map[i]: means[i].tolist()
                for i in range(self.n_states)
            },
            "training_samples": features.shape[0],
            "converged": self.model.monitor_.converged,
        }

    def _label_states(self, means: np.ndarray) -> dict[int, str]:
        """
        Auto-label HMM states based on feature characteristics:
        - crisis: highest VIX z-score + highest credit spread
        - risk_off: moderate VIX + negative SPY returns
        - risk_on: lowest VIX + positive SPY returns
        - normal: everything else
        """
        scores = []
        for i in range(self.n_states):
            vix_z = means[i][0]       # vix_zscore
            spy_ret = means[i][2]     # spy_return_5d
            credit = means[i][4] if means.shape[1] > 4 else 0  # credit_spread
            # Higher score = more stressed
            stress = vix_z * 2 + credit * 0.5 - spy_ret * 1.5
            scores.append((i, stress, vix_z, spy_ret))

        scores.sort(key=lambda x: x[1], reverse=True)

        labels = {}
        labels[scores[0][0]] = "crisis"
        labels[scores[1][0]] = "risk_off"
        labels[scores[-1][0]] = "risk_on"
        # Remaining state is "normal"
        for i in range(self.n_states):
            if i not in labels:
                labels[i] = "normal"

        return labels

    def predict(self, features: np.ndarray) -> dict:
        """
        Predict regime state for current feature vector(s).

        Args:
            features: shape (1, 8) or (T, 8)

        Returns:
            dict with state, probabilities, confidence, raw_state
        """
        if self.model is None:
            return {
                "state": "normal",
                "probabilities": {s: 0.25 for s in STATE_LABELS},
                "confidence": 0.0,
                "error": "Model not trained — call /regime/train first",
            }

        # Scrub input
        features = _scrub_features(features)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Validate feature dimensions match trained model
        expected_dim = self.model.means_.shape[1]
        if features.shape[1] != expected_dim:
            return {
                "state": "normal",
                "probabilities": {s: 0.25 for s in STATE_LABELS},
                "confidence": 0.0,
                "error": f"Feature dimension mismatch: got {features.shape[1]}, expected {expected_dim}",
            }

        try:
            # Get state probabilities for the last observation
            posteriors = self.model.predict_proba(features)
            last_posterior = posteriors[-1]

            raw_state = int(np.argmax(last_posterior))
            state_label = self.state_map.get(raw_state, "normal")
            confidence = float(last_posterior[raw_state]) * 100

            probs = {
                self.state_map.get(i, f"state_{i}"): round(float(last_posterior[i]) * 100, 2)
                for i in range(self.n_states)
            }

            return {
                "state": state_label,
                "probabilities": probs,
                "confidence": round(confidence, 2),
                "raw_state": raw_state,
                "transition_matrix": self.model.transmat_.tolist(),
            }

        except Exception as e:
            log.error(f"HMM predict failed: {e}")
            return {
                "state": "normal",
                "probabilities": {s: 0.25 for s in STATE_LABELS},
                "confidence": 0.0,
                "error": f"Prediction failed: {str(e)}",
            }


# Singleton instance
_hmm = RegimeHMM()


def get_hmm() -> RegimeHMM:
    return _hmm


def get_hmm() -> RegimeHMM:
    return _hmm
