"""Independent model artifact loading, inference, and engineering explanations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.feature_extraction import MODEL_FEATURE_COLUMNS
from src.preprocessing import validate_feature_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed_features.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "dataset_configurations.csv"
DEFAULT_JOURNAL_PATH = PROJECT_ROOT / "data" / "journal_documented_results.json"
DEFAULT_BOUNDARIES_DIR = PROJECT_ROOT / "data" / "stability_boundaries"
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def artifacts_exist() -> bool:
    return DEFAULT_MODEL_PATH.is_file() and DEFAULT_METADATA_PATH.is_file()


def load_model_bundle() -> Tuple[Any, Dict[str, Any]]:
    """Load saved scikit-learn Pipeline and model metadata independently of notebooks."""
    if not artifacts_exist():
        raise FileNotFoundError(
            "Trained model artifacts not found in `models/`. "
            "Run `python train_pipeline.py` locally with your HDF5 folders to generate "
            "`models/final_model.joblib` and `models/model_metadata.json`."
        )
    model = joblib.load(DEFAULT_MODEL_PATH)
    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def load_evaluation_results() -> Optional[Dict[str, Any]]:
    if not DEFAULT_EVAL_PATH.is_file():
        return None
    with open(DEFAULT_EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_journal_benchmarks() -> Dict[str, Any]:
    if not DEFAULT_JOURNAL_PATH.is_file():
        return {}
    with open(DEFAULT_JOURNAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed_dataset() -> Optional[pd.DataFrame]:
    if not DEFAULT_DATA_PATH.is_file():
        return None
    return pd.read_csv(DEFAULT_DATA_PATH)


def load_dataset_configurations() -> pd.DataFrame:
    if not DEFAULT_CONFIG_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(DEFAULT_CONFIG_PATH)


def predict_single(
    model: Any,
    feature_dict: Dict[str, float],
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run ML inference for a single vibration feature vector."""
    cols = feature_columns or MODEL_FEATURE_COLUMNS
    row_df = pd.DataFrame([{c: float(feature_dict[c]) for c in cols}])
    pred_class = int(model.predict(row_df)[0])

    prob_stable: Optional[float] = None
    prob_chatter: Optional[float] = None
    confidence: Optional[float] = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row_df)[0]
        classes = list(model.classes_)
        if 0 in classes and 1 in classes:
            prob_stable = float(probs[classes.index(0)])
            prob_chatter = float(probs[classes.index(1)])
            confidence = float(max(prob_stable, prob_chatter))

    return {
        "predicted_class": pred_class,
        "predicted_label": "CHATTER" if pred_class == 1 else "STABLE",
        "prob_stable": prob_stable,
        "prob_chatter": prob_chatter,
        "confidence": confidence,
        "features_used": {c: float(feature_dict[c]) for c in cols},
    }


def predict_batch(
    model: Any,
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and predict on a batch DataFrame of vibration features."""
    cols = feature_columns or MODEL_FEATURE_COLUMNS
    clean_df, missing = validate_feature_dataframe(df, cols)
    if missing:
        return clean_df, missing

    X = clean_df[cols]
    preds = model.predict(X).astype(int)
    out = clean_df.copy()
    out["Predicted_Class"] = preds
    out["Predicted_Label"] = np.where(preds == 1, "CHATTER", "STABLE")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        if 0 in classes and 1 in classes:
            out["Prob_Stable"] = probs[:, classes.index(0)]
            out["Prob_Chatter"] = probs[:, classes.index(1)]
            out["Confidence"] = np.maximum(out["Prob_Stable"], out["Prob_Chatter"])

    return out, []


def generate_engineering_explanation(
    prediction: Dict[str, Any],
    metadata: Dict[str, Any],
    physics_result: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Produce concise, grounded engineering interpretation of a prediction."""
    notes: List[str] = []
    feats = prediction["features_used"]
    stats = metadata.get("class_feature_medians", {})
    stable_med = stats.get("0", {})
    chatter_med = stats.get("1", {})

    rms = feats.get("RMS", 0.0)
    p2p = feats.get("Peak-to-Peak", 0.0)
    max_fft = feats.get("Max FFT", 0.0)
    dom_freq = feats.get("Dominant Frequency", 0.0)

    if prediction["predicted_class"] == 1:
        notes.append(
            f"**Elevated Vibration Energy:** The classifier flagged **CHATTER** "
            f"(RMS = `{rms:.2f} m/s^2`, Peak-to-Peak = `{p2p:.2f} m/s^2`, Max FFT = `{max_fft:.2f} m/s^2`), "
            "consistent with self-excited regenerative vibration growth."
        )
    else:
        notes.append(
            f"**Controlled Vibration Profile:** The classifier predicted **STABLE** cutting "
            f"(RMS = `{rms:.2f} m/s^2`, Peak-to-Peak = `{p2p:.2f} m/s^2`, Max FFT = `{max_fft:.2f} m/s^2`), "
            "indicating forced tooth-passing vibration without runaway chatter amplification."
        )

    if stable_med and chatter_med and "RMS" in stable_med and "RMS" in chatter_med:
        s_rms = stable_med["RMS"]
        c_rms = chatter_med["RMS"]
        ratio_s = (rms / s_rms) if s_rms > 0 else 0.0
        notes.append(
            f"**Reference Scale Comparison:** Input RMS is `{ratio_s:.2f}x` the training Stable median "
            f"(`{s_rms:.2f} m/s^2`) vs. the training Chatter median (`{c_rms:.2f} m/s^2`)."
        )

    notes.append(
        f"**Spectral Signature:** Dominant vibration energy is concentrated at `{dom_freq:.1f} Hz`. "
        "Note that if the test configuration has a substantially different natural frequency "
        "(e.g., 1500 Hz vs. 500 Hz training systems), cross-machine domain shift can reduce reliability."
    )

    if physics_result is not None:
        agree = physics_result["physics_label"] == prediction["predicted_class"]
        verdict = "AGREES" if agree else "DISAGREES"
        notes.append(
            f"**Physics Stability Boundary Check ({verdict}):** At `{physics_result['omega']:.0f} rpm`, "
            f"analytical critical depth is `{physics_result['critical_depth']*1000:.3f} mm` vs. "
            f"actual depth `{physics_result['depth']*1000:.3f} mm` "
            f"(analytical state: **{physics_result['physics_state']}**, margin `{physics_result['margin_pct']:+.1f}%`)."
        )

    return notes
