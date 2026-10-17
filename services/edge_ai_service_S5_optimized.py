import os
import json
import hashlib
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import joblib
import pandas as pd
import sklearn
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Paths / configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "parking_status_model.joblib"
FEATURE_ORDER_PATH = MODELS_DIR / "feature_order.json"

EXECUTION_SITE = os.getenv("EXECUTION_SITE", "edge")


# ============================================================
# Request schema
# ============================================================

class PredictNamedRequest(BaseModel):
    # Any allows optional future metadata while the six required
    # AI features are explicitly validated below.
    features: Dict[str, Any]


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_feature_order(path: Path) -> List[str]:
    if not path.exists():
        raise RuntimeError(f"Feature-order file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        order = obj
    elif isinstance(obj, dict) and isinstance(obj.get("feature_order"), list):
        order = obj["feature_order"]
    else:
        raise RuntimeError("Invalid feature_order.json format")

    if not order or not all(isinstance(x, str) for x in order):
        raise RuntimeError("feature_order.json does not contain valid feature names")

    return order


# ============================================================
# Load and verify frozen artifact
# ============================================================

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH}")

FEATURE_ORDER = load_feature_order(FEATURE_ORDER_PATH)

ARTIFACT = joblib.load(MODEL_PATH)

if isinstance(ARTIFACT, dict):
    MODEL = ARTIFACT.get("model")
    ARTIFACT_FEATURE_ORDER = ARTIFACT.get("feature_order")
    ARTIFACT_LABELS = ARTIFACT.get("labels")
else:
    MODEL = ARTIFACT
    ARTIFACT_FEATURE_ORDER = None
    ARTIFACT_LABELS = None

if MODEL is None or not hasattr(MODEL, "predict"):
    raise RuntimeError("Loaded artifact does not contain a usable classifier")

N_FEATURES_EXPECTED = int(getattr(MODEL, "n_features_in_", len(FEATURE_ORDER)))

if N_FEATURES_EXPECTED != len(FEATURE_ORDER):
    raise RuntimeError(
        f"Model expects {N_FEATURES_EXPECTED} features but "
        f"feature_order.json contains {len(FEATURE_ORDER)}"
    )

# Artifact contract must match external JSON contract.
if ARTIFACT_FEATURE_ORDER is not None:
    if list(ARTIFACT_FEATURE_ORDER) != list(FEATURE_ORDER):
        raise RuntimeError(
            "Feature-order mismatch between model artifact and feature_order.json"
        )

# If sklearn stored feature names, verify those too.
MODEL_FEATURE_NAMES = getattr(MODEL, "feature_names_in_", None)

if MODEL_FEATURE_NAMES is not None:
    if list(MODEL_FEATURE_NAMES) != list(FEATURE_ORDER):
        raise RuntimeError(
            "Feature-order mismatch between trained model and feature_order.json"
        )

MODEL_SHA256 = sha256_file(MODEL_PATH)

# ============================================================
# Edge inference runtime policy
# ============================================================

# The frozen model artifact on disk is not modified.
# Restrict only the loaded runtime classifier to one worker
# to avoid nested Random-Forest CPU oversubscription.
if hasattr(MODEL, "n_jobs"):
    MODEL.n_jobs = 1

# FastAPI synchronous endpoints may execute in worker threads.
# Serialize model evaluation so local inference requests cannot
# overlap and create CPU-contention cascades.
INFERENCE_LOCK = threading.Lock()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Smart Parking Edge AI Service",
    version="2.1.0-s5-optimized"
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "smart-parking-edge-ai",
        "execution_site": EXECUTION_SITE,
        "hint": "Use /health or /predict_named",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "execution_site": EXECUTION_SITE,
        "model_path": str(MODEL_PATH),
        "model_sha256": MODEL_SHA256,
        "n_features_expected": N_FEATURES_EXPECTED,
        "feature_order": FEATURE_ORDER,
        "classes": (
            [str(x) for x in MODEL.classes_]
            if hasattr(MODEL, "classes_")
            else []
        ),
        "n_estimators": getattr(MODEL, "n_estimators", None),
        "model_runtime_n_jobs": getattr(MODEL, "n_jobs", None),
        "inference_strategy": "single_predict_proba_argmax",
        "inference_serialized": True,
        "python_model_runtime": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


@app.post("/predict_named")
def predict_named(req: PredictNamedRequest):
    missing = [
        name
        for name in FEATURE_ORDER
        if name not in req.features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}"
        )

    try:
        row = {
            name: float(req.features[name])
            for name in FEATURE_ORDER
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"All required features must be numeric: {exc}"
        )

    # Use named DataFrame columns exactly as used during training.
    X = pd.DataFrame(
        [[row[name] for name in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    t0 = perf_counter()

    # Include any serialization wait in service_inference_ms so
    # queueing delay is visible rather than hidden.
    with INFERENCE_LOCK:
        if not hasattr(MODEL, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="Classifier does not expose predict_proba"
            )

        # Exactly one Random-Forest evaluation per request.
        p = MODEL.predict_proba(X)[0]

        classes = (
            MODEL.classes_.tolist()
            if hasattr(MODEL, "classes_")
            else list(range(len(p)))
        )

        prediction = str(
            classes[int(np.argmax(p))]
        ).upper()

        probabilities = {
            str(label): float(prob)
            for label, prob in zip(classes, p)
        }

    inference_ms = (perf_counter() - t0) * 1000.0

    return {
        # Original API-compatible fields
        "prediction": prediction,
        "prediction_type": "str",
        "n_features_expected": N_FEATURES_EXPECTED,
        "feature_order": FEATURE_ORDER,
        "x": [row[name] for name in FEATURE_ORDER],
        "probabilities": probabilities,

        # New experimental metadata
        "execution_site": EXECUTION_SITE,
        "service_inference_ms": round(inference_ms, 6),
        "model_sha256": MODEL_SHA256,
    }
