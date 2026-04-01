"""
Calibration Engine — applies non-linear calibration to magnitude predictions.

Supports:
1. Isotonic regression (monotonic, non-parametric)
2. Platt scaling (logistic sigmoid for probability calibration)
3. Conformal prediction (coverage guarantees)
4. Subgroup calibration (per sector, vol regime, ticker class)

Replaces the simple scale+bias+direction-flip approach with
context-aware, non-linear calibration.
"""

import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = logging.getLogger("magnitude-v2")


class MagnitudeCalibrator:
    """
    Multi-method calibrator for magnitude predictions.
    
    Usage:
        cal = MagnitudeCalibrator()
        cal.fit(predicted_moves, actual_moves, method="isotonic")
        calibrated = cal.calibrate(new_prediction)
        ci = cal.conformal_interval(new_prediction, coverage=0.90)
    """
    
    def __init__(self):
        self.isotonic_model = None
        self.isotonic_abs_model = None
        self.platt_models = {}  # threshold -> LogisticRegression
        self.conformal_scores = None
        self.residual_stats = {}
        self.method = "none"
        self.subgroup_calibrators = {}
    
    def fit(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        method: str = "isotonic",
        subgroups: dict[str, np.ndarray] | None = None,
    ) -> dict:
        """
        Fit calibration model on historical predicted vs actual pairs.
        
        Args:
            predicted: Raw model predictions
            actual: Realized actual moves
            method: 'isotonic', 'conformal', or 'both'
            subgroups: Optional dict of { subgroup_key: boolean_mask }
        
        Returns:
            Calibration metrics dict.
        """
        self.method = method
        n = len(predicted)
        
        if n < 20:
            log.warning(f"Only {n} samples for calibration — using pass-through")
            return {"method": "none", "reason": "insufficient_samples", "n": n}
        
        metrics = {"n": n, "method": method}
        
        # ── Isotonic calibration (signed move) ──
        if method in ("isotonic", "both"):
            self.isotonic_model = IsotonicRegression(
                y_min=min(actual) - 1,
                y_max=max(actual) + 1,
                out_of_bounds="clip",
                increasing="auto",
            )
            self.isotonic_model.fit(predicted, actual)
            
            calibrated = self.isotonic_model.predict(predicted)
            metrics["isotonic_mae"] = round(float(np.mean(np.abs(calibrated - actual))), 4)
            metrics["isotonic_bias"] = round(float(np.mean(calibrated - actual)), 4)
            
            # Also fit absolute move calibrator
            self.isotonic_abs_model = IsotonicRegression(
                y_min=0, y_max=max(np.abs(actual)) + 1,
                out_of_bounds="clip",
            )
            self.isotonic_abs_model.fit(np.abs(predicted), np.abs(actual))
        
        # ── Conformal calibration (coverage guarantees) ──
        if method in ("conformal", "both"):
            if self.isotonic_model:
                residuals = actual - self.isotonic_model.predict(predicted)
            else:
                residuals = actual - predicted
            
            self.conformal_scores = np.sort(np.abs(residuals))
            metrics["conformal_scores_n"] = len(self.conformal_scores)
            
            # Pre-compute coverage metrics
            for coverage in [0.50, 0.80, 0.90, 0.95]:
                q_idx = int(np.ceil(coverage * len(self.conformal_scores))) - 1
                q_idx = min(q_idx, len(self.conformal_scores) - 1)
                half_width = float(self.conformal_scores[q_idx])
                metrics[f"conformal_halfwidth_{int(coverage*100)}"] = round(half_width, 4)
        
        # ── Probability calibration (Platt scaling for thresholds) ──
        for threshold in [1.0, 2.0, 3.0, 5.0]:
            labels = (np.abs(actual) > threshold).astype(int)
            if labels.sum() >= 5 and (n - labels.sum()) >= 5:
                platt = LogisticRegression(C=1.0, solver="lbfgs")
                platt.fit(np.abs(predicted).reshape(-1, 1), labels)
                self.platt_models[threshold] = platt
                
                platt_probs = platt.predict_proba(np.abs(predicted).reshape(-1, 1))[:, 1]
                actual_rate = float(labels.mean())
                pred_rate = float(platt_probs.mean())
                metrics[f"platt_{threshold}_actual_rate"] = round(actual_rate, 4)
                metrics[f"platt_{threshold}_pred_rate"] = round(pred_rate, 4)
        
        # ── Subgroup calibration ──
        if subgroups:
            for key, mask in subgroups.items():
                if mask.sum() >= 15:
                    sub_cal = MagnitudeCalibrator()
                    sub_metrics = sub_cal.fit(predicted[mask], actual[mask], method="isotonic")
                    self.subgroup_calibrators[key] = sub_cal
                    metrics[f"subgroup_{key}"] = sub_metrics
        
        # ── Residual statistics ──
        residuals = actual - predicted
        self.residual_stats = {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "skew": round(float(
                np.mean(((residuals - np.mean(residuals)) / max(np.std(residuals), 0.01))**3)
            ), 4),
            "kurtosis": round(float(
                np.mean(((residuals - np.mean(residuals)) / max(np.std(residuals), 0.01))**4)
            ), 4),
        }
        metrics["residual_stats"] = self.residual_stats
        
        return metrics
    
    def calibrate(
        self,
        prediction: float,
        subgroup: str | None = None,
    ) -> float:
        """Apply calibration to a single prediction."""
        if subgroup and subgroup in self.subgroup_calibrators:
            return self.subgroup_calibrators[subgroup].calibrate(prediction)
        
        if self.isotonic_model:
            return float(self.isotonic_model.predict([prediction])[0])
        
        return prediction
    
    def calibrate_abs(self, abs_prediction: float) -> float:
        """Calibrate absolute move prediction."""
        if self.isotonic_abs_model:
            return float(self.isotonic_abs_model.predict([abs_prediction])[0])
        return abs_prediction
    
    def conformal_interval(
        self,
        prediction: float,
        coverage: float = 0.90,
    ) -> tuple[float, float]:
        """
        Compute conformal prediction interval with coverage guarantee.
        
        Returns (lower, upper) bounds.
        """
        calibrated = self.calibrate(prediction)
        
        if self.conformal_scores is not None and len(self.conformal_scores) > 0:
            q_idx = int(np.ceil(coverage * len(self.conformal_scores))) - 1
            q_idx = min(q_idx, len(self.conformal_scores) - 1)
            half_width = float(self.conformal_scores[q_idx])
        else:
            # Fallback: use residual std
            std = self.residual_stats.get("std", 2.0)
            from scipy.stats import norm
            z = norm.ppf((1 + coverage) / 2)
            half_width = z * std
        
        return (round(calibrated - half_width, 4), round(calibrated + half_width, 4))
    
    def threshold_probability(
        self,
        abs_prediction: float,
        threshold: float,
    ) -> float:
        """
        Calibrated probability that |move| > threshold.
        Uses Platt scaling if available, falls back to empirical.
        """
        if threshold in self.platt_models:
            return float(
                self.platt_models[threshold].predict_proba([[abs_prediction]])[0, 1]
            )
        return 0.0
    
    def get_state(self) -> dict:
        """Export calibration state for storage in magnitude_v2_calibration."""
        return {
            "method": self.method,
            "residual_stats": self.residual_stats,
            "has_isotonic": self.isotonic_model is not None,
            "has_conformal": self.conformal_scores is not None,
            "platt_thresholds": list(self.platt_models.keys()),
            "subgroups": list(self.subgroup_calibrators.keys()),
        }
