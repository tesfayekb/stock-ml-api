"""
Magnitude v2 — Multi-specialist magnitude prediction system.

Replaces heuristic blending with learned models, quantile regression,
conformal calibration, and disagreement-aware uncertainty.
"""

from .feature_store import extract_features, FEATURE_VERSION
from .baseline_model import train_baseline_model, predict_baseline
from .earnings_model import train_earnings_model
from .event_model import train_event_model
from .meta_model import (
    blend_predictions,
    compute_dynamic_weights,
    compute_disagreement,
    SpecialistOutput,
    DisagreementResult,
)
from .calibration import MagnitudeCalibrator

__all__ = [
    "extract_features",
    "FEATURE_VERSION",
    "train_baseline_model",
    "predict_baseline",
    "train_earnings_model",
    "train_event_model",
    "blend_predictions",
    "compute_dynamic_weights",
    "compute_disagreement",
    "SpecialistOutput",
    "DisagreementResult",
    "MagnitudeCalibrator",
]
