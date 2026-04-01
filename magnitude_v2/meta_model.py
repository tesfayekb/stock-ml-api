"""
Meta-Model — selects and blends specialist model outputs based on
recent performance, regime context, disagreement, and data completeness.

This is the decision layer that produces the final magnitude v2 prediction.
"""

import numpy as np
import logging
from dataclasses import dataclass

log = logging.getLogger("magnitude-v2")


@dataclass
class SpecialistOutput:
    specialist_type: str
    expected_move_pct: float
    expected_abs_move_pct: float
    direction_confidence: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    model_confidence: float
    training_samples: int
    path_class: str = "drift"
    path_confidence: float = 0.0
    prob_move_gt_1pct: float = 0.0
    prob_move_gt_2pct: float = 0.0
    prob_move_gt_3pct: float = 0.0
    prob_move_gt_5pct: float = 0.0


@dataclass
class DisagreementResult:
    specialist_disagreement: float
    scanner_ml_disagreement: float
    thesis_ml_disagreement: float
    direction_disagreement: bool
    action: str  # 'none', 'widen_ci', 'reduce_certainty', 'abstain'
    ci_widening_factor: float
    certainty_downgrade: str | None


def compute_disagreement(
    specialists: list[SpecialistOutput],
    scanner_prediction: float | None = None,
    thesis_prediction: float | None = None,
) -> DisagreementResult:
    """
    Compute disagreement metrics across specialist models and external sources.
    
    Disagreement is one of the strongest confidence signals — high disagreement
    means wider CIs and lower certainty.
    """
    if len(specialists) < 2:
        return DisagreementResult(
            specialist_disagreement=0, scanner_ml_disagreement=0,
            thesis_ml_disagreement=0, direction_disagreement=False,
            action="none", ci_widening_factor=1.0, certainty_downgrade=None,
        )
    
    # Specialist disagreement: std of point predictions
    moves = [s.expected_move_pct for s in specialists]
    specialist_disagreement = float(np.std(moves))
    
    # Direction disagreement: any specialists disagree on direction?
    signs = [np.sign(s.expected_move_pct) for s in specialists if abs(s.expected_move_pct) > 0.1]
    direction_disagreement = len(set(signs)) > 1 if signs else False
    
    # Scanner disagreement
    blended_move = float(np.mean(moves))
    scanner_ml_disagreement = 0.0
    if scanner_prediction is not None:
        scanner_ml_disagreement = abs(scanner_prediction - blended_move)
    
    # Thesis disagreement
    thesis_ml_disagreement = 0.0
    if thesis_prediction is not None:
        thesis_ml_disagreement = abs(thesis_prediction - blended_move)
    
    # Determine action based on disagreement severity
    total_disagreement = specialist_disagreement + scanner_ml_disagreement * 0.5 + thesis_ml_disagreement * 0.3
    
    if total_disagreement > 3.0 or direction_disagreement:
        action = "abstain"
        ci_widening = 2.0
        certainty_downgrade = "low"
    elif total_disagreement > 2.0:
        action = "reduce_certainty"
        ci_widening = 1.5
        certainty_downgrade = "moderate" if specialist_disagreement > 1.5 else None
    elif total_disagreement > 1.0:
        action = "widen_ci"
        ci_widening = 1.2
        certainty_downgrade = None
    else:
        action = "none"
        ci_widening = 1.0
        certainty_downgrade = None
    
    return DisagreementResult(
        specialist_disagreement=round(specialist_disagreement, 4),
        scanner_ml_disagreement=round(scanner_ml_disagreement, 4),
        thesis_ml_disagreement=round(thesis_ml_disagreement, 4),
        direction_disagreement=direction_disagreement,
        action=action,
        ci_widening_factor=ci_widening,
        certainty_downgrade=certainty_downgrade,
    )


def compute_dynamic_weights(
    specialists: list[SpecialistOutput],
    calibration_metrics: dict,
    regime_state: str = "unknown",
    has_earnings: bool = False,
    has_catalyst: bool = False,
) -> dict[str, float]:
    """
    Compute dynamic specialist weights based on recent performance,
    regime context, and event presence.
    
    Replaces fixed blending ratios (65/35, 40/60) with learned weights.
    """
    weights = {}
    
    for s in specialists:
        # Base weight from model confidence (correlation)
        base_w = max(0.1, min(1.0, abs(s.model_confidence)))
        
        # Scale by training sample count (trust models with more data)
        sample_factor = min(1.0, s.training_samples / 200) if s.training_samples > 0 else 0.3
        
        # Calibration performance adjustment
        cal = calibration_metrics.get(s.specialist_type, {})
        cal_mae = cal.get("mae", 999)
        cal_factor = max(0.3, min(1.5, 2.0 / max(cal_mae, 0.5)))
        
        weights[s.specialist_type] = base_w * sample_factor * cal_factor
    
    # Context overrides
    if has_earnings and "earnings" in weights:
        weights["earnings"] *= 2.5  # Strongly prefer earnings specialist
        weights["baseline"] = weights.get("baseline", 0) * 0.3
        weights["event"] = weights.get("event", 0) * 0.4
    
    elif has_catalyst and "event" in weights:
        weights["event"] *= 1.8
        weights["baseline"] = weights.get("baseline", 0) * 0.6
    
    # Regime adjustments
    if regime_state in ("high_volatility", "risk_off", "choppy"):
        # In volatile regimes, trust baseline more (it sees vol features)
        weights["baseline"] = weights.get("baseline", 0) * 1.3
    
    # Normalize to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    
    return weights


def blend_predictions(
    specialists: list[SpecialistOutput],
    weights: dict[str, float],
    disagreement: DisagreementResult,
    regime_state: str = "unknown",
    vol_regime: str = "normal",
) -> dict:
    """
    Produce final blended magnitude prediction from specialist outputs.
    
    Returns dict matching magnitude_v2_predictions schema.
    """
    if not specialists:
        return {"status": "no_specialists"}
    
    # ── Weighted point prediction ──
    expected_move = 0.0
    expected_abs_move = 0.0
    
    for s in specialists:
        w = weights.get(s.specialist_type, 0)
        expected_move += s.expected_move_pct * w
        expected_abs_move += s.expected_abs_move_pct * w
    
    # ── Weighted quantile blending ──
    blended_q = {}
    for q_name in ["q10", "q25", "q50", "q75", "q90"]:
        q_val = 0.0
        for s in specialists:
            w = weights.get(s.specialist_type, 0)
            q_val += getattr(s, q_name, 0) * w
        blended_q[q_name] = round(q_val, 4)
    
    # Apply disagreement CI widening
    if disagreement.ci_widening_factor > 1.0:
        q_center = blended_q["q50"]
        for q_name in ["q10", "q25", "q75", "q90"]:
            spread = blended_q[q_name] - q_center
            blended_q[q_name] = round(q_center + spread * disagreement.ci_widening_factor, 4)
    
    ci_low = blended_q["q10"]
    ci_high = blended_q["q90"]
    
    # ── Threshold probabilities (weighted average) ──
    threshold_probs = {}
    for thresh_key in ["prob_move_gt_1pct", "prob_move_gt_2pct", "prob_move_gt_3pct", "prob_move_gt_5pct"]:
        prob = 0.0
        for s in specialists:
            w = weights.get(s.specialist_type, 0)
            prob += getattr(s, thresh_key, 0) * w
        threshold_probs[thresh_key] = round(prob, 4)
    
    # ── Path classification (majority vote weighted by confidence) ──
    path_votes = {}
    for s in specialists:
        if s.path_class:
            w = weights.get(s.specialist_type, 0) * max(0.1, s.path_confidence)
            path_votes[s.path_class] = path_votes.get(s.path_class, 0) + w
    
    path_class = max(path_votes, key=path_votes.get) if path_votes else "drift"
    path_total = sum(path_votes.values())
    path_confidence = path_votes.get(path_class, 0) / path_total if path_total > 0 else 0
    
    # ── Certainty level ──
    avg_confidence = sum(s.model_confidence * weights.get(s.specialist_type, 0) for s in specialists)
    ci_width = ci_high - ci_low
    
    if disagreement.certainty_downgrade:
        certainty_level = disagreement.certainty_downgrade
    elif avg_confidence > 0.3 and ci_width < 3.0:
        certainty_level = "very_high"
    elif avg_confidence > 0.2 and ci_width < 5.0:
        certainty_level = "high"
    elif avg_confidence > 0.1:
        certainty_level = "moderate"
    else:
        certainty_level = "low"
    
    # ── Winning specialist ──
    winning = max(weights, key=weights.get) if weights else None
    
    return {
        "expected_move_pct": round(expected_move, 4),
        "expected_abs_move_pct": round(expected_abs_move, 4),
        "ci_low": ci_low,
        "ci_high": ci_high,
        **blended_q,
        **threshold_probs,
        "path_class": path_class,
        "path_confidence": round(path_confidence, 4),
        "certainty_level": certainty_level,
        "meta_confidence": round(avg_confidence, 4),
        "specialist_weights": weights,
        "winning_specialist": winning,
        "specialist_disagreement": disagreement.specialist_disagreement,
        "scanner_ml_disagreement": disagreement.scanner_ml_disagreement,
        "thesis_ml_disagreement": disagreement.thesis_ml_disagreement,
        "disagreement_action": disagreement.action,
        "regime_state": regime_state,
        "vol_regime": vol_regime,
    }
