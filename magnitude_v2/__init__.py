"""
Magnitude v2 — Multi-specialist magnitude prediction system.

Replaces heuristic blending with learned models, quantile regression,
conformal calibration, and disagreement-aware uncertainty.

v15e: Added model persistence layer (save/load to Supabase Storage)
for stateless Railway container support.
"""

import io
import logging
from datetime import datetime

import joblib

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

log = logging.getLogger("magnitude_v2")


# ── v15e: Model Persistence Layer ────────────────────────────────────────

def save_model_artifact(sb_client, specialist_type: str, ticker: str,
                        model_dict: dict, version: str, user_id: str) -> dict:
    """
    Serialize and upload a trained specialist model to Supabase Storage.

    Path format: magnitude_v2/{specialist_type}/{ticker}/{version}.joblib
    Also updates magnitude_v2_calibration with model_stored_at and model_version.

    Args:
        sb_client: Supabase client instance
        specialist_type: "baseline", "earnings", or "event"
        ticker: Stock ticker (e.g. "AAPL")
        model_dict: Dict with point_model, quantile_models, feature_names, etc.
        version: Version string (e.g. "v2.0-202604041530")
        user_id: User ID for calibration table update

    Returns:
        Dict with path, size_bytes, stored_at
    """
    storage_path = f"magnitude_v2/{specialist_type}/{ticker}/{version}.joblib"

    # Serialize model to bytes
    buffer = io.BytesIO()
    artifact = {
        "point_model": model_dict.get("point_model"),
        "quantile_models": model_dict.get("quantile_models"),
        "feature_names": model_dict.get("feature_names") or model_dict.get("features_used"),
        "model_version": version,
        "specialist_type": specialist_type,
        "ticker": ticker,
        "trained_at": datetime.utcnow().isoformat(),
    }
    joblib.dump(artifact, buffer)
    model_bytes = buffer.getvalue()

    # Upload to Supabase Storage
    sb_client.storage.from_("ml-models").upload(
        storage_path,
        model_bytes,
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )

    # Update calibration row with storage metadata
    sb_client.table("magnitude_v2_calibration").update({
        "model_stored_at": datetime.utcnow().isoformat(),
        "model_version": version,
    }).eq("user_id", user_id).eq("specialist_type", specialist_type).eq(
        "subgroup_key", ticker
    ).execute()

    log.info(f"Model stored: {storage_path} ({len(model_bytes)} bytes)")

    return {
        "path": storage_path,
        "size_bytes": len(model_bytes),
        "stored_at": datetime.utcnow().isoformat(),
    }


def load_model_artifact(sb_client, specialist_type: str, ticker: str,
                        user_id: str, version: str = "latest") -> dict | None:
    """
    Download and deserialize a specialist model from Supabase Storage.

    If version=="latest", queries magnitude_v2_calibration for the most recent
    model_stored_at row to determine the version.

    Args:
        sb_client: Supabase client instance
        specialist_type: "baseline", "earnings", or "event"
        ticker: Stock ticker
        user_id: User ID for calibration lookup
        version: Specific version or "latest"

    Returns:
        Model dict (point_model, quantile_models, feature_names, ...) or None
    """
    if version == "latest":
        # Look up latest version from calibration table
        resp = sb_client.table("magnitude_v2_calibration") \
            .select("model_version, model_stored_at") \
            .eq("user_id", user_id) \
            .eq("specialist_type", specialist_type) \
            .eq("subgroup_key", ticker) \
            .not_.is_("model_stored_at", "null") \
            .order("model_stored_at", desc=True) \
            .limit(1) \
            .execute()

        if not resp.data:
            return None
        version = resp.data[0]["model_version"]

    storage_path = f"magnitude_v2/{specialist_type}/{ticker}/{version}.joblib"

    try:
        data = sb_client.storage.from_("ml-models").download(storage_path)
        if not data:
            log.warning(f"Empty download for {storage_path}")
            return None

        buffer = io.BytesIO(data)
        artifact = joblib.load(buffer)
        log.info(f"Model loaded from storage: {storage_path}")
        return artifact
    except Exception as e:
        log.warning(f"Model load failed for {storage_path}: {e}")
        return None


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
    "save_model_artifact",
    "load_model_artifact",
]
