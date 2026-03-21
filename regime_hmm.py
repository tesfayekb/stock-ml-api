"""
Gaussian HMM Regime Classifier — 4-state regime detection.

States: risk_on, normal, risk_off, crisis
Features (8D): vix_zscore, vix_term_ratio, spy_return_5d, spy_return_20d,
               credit_spread, yield_curve_slope, cross_asset_momentum, dollar_strength
"""

import numpy as np
import pickle
import os
from hmmlearn.hmm import GaussianHMM

MODEL_PATH = os.environ.get("HMM_MODEL_PATH", "/tmp/regime_hmm.pkl")
STATE_LABELS = ["risk_on", "normal", "risk_off", "crisis"]

class RegimeHMM:
    """Wrapper around hmmlearn GaussianHMM with auto state labeling."""

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model: GaussianHMM | None = None
        self.state_map: dict[int, str] = {}
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
                self.model = saved["model"]
                self.state_map = saved["state_map"]

    def _save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "state_map": self.state_map}, f)

    def train(self, features: np.ndarray, n_iter: int = 200) -> dict:
        """
        Train HMM on historical feature matrix.

        Args:
            features: shape (T, 8) — standardized feature matrix
            n_iter: EM iterations

        Returns:
            dict with state_labels, transition_matrix, means, training_samples
        """
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )
        self.model.fit(features)

        # Auto-label states by analyzing emission means
        # Feature 0 = vix_zscore, Feature 2 = spy_return_5d, Feature 4 = credit_spread
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
            credit = means[i][4]      # credit_spread
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
                "confidence": 25.0,
                "error": "Model not trained",
            }

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get state probabilities for the last observation
        log_probs = self.model.score_samples(features)
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


# Singleton instance
_hmm = RegimeHMM()


def get_hmm() -> RegimeHMM:
    return _hmm
```

## Endpoints to add in `main.py`

```python
from regime_hmm import get_hmm
import numpy as np

@app.post("/regime/train")
async def regime_train(request: Request):
    """Train HMM on historical feature matrix from bootstrap function."""
    body = await request.json()
    features = np.array(body["features"], dtype=np.float64)
    n_states = body.get("n_states", 4)

    hmm = get_hmm()
    if n_states != hmm.n_states:
        hmm.n_states = n_states
        hmm.model = None

    result = hmm.train(features)
    return {"success": True, **result}


@app.post("/regime/predict")
async def regime_predict(request: Request):
    """Predict current regime from feature vector."""
    body = await request.json()
    features = np.array(body["features"], dtype=np.float64)

    hmm = get_hmm()
    result = hmm.predict(features)
    return {"success": True, **result}
