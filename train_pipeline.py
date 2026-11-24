"""Reproducible Phase 2 ML Training & Validation Pipeline for Milling Chatter Detection.

Runs the corrected research methodology from JOURNAL.ipynb:
  1. Extracts validated DSP features (detrend + Hanning + normalized rFFT in Hz) from HDF5 files
     OR loads an existing validated CSV.
  2. Computes physics-based chatter labels using each dataset's own stability_boundaryX.h5.
  3. Strictly separates vibration ML inputs (`RMS`, `Peak-to-Peak`, `Max FFT`, `Dominant Frequency`)
     from experiment metadata (`Dataset`, `Omega`, `Depth`, `Critical Depth`).
  4. Compares Logistic Regression, Decision Tree, and Random Forest using:
     - Leave-One-Dataset-Out (LODO) cross-validation (Primary: unseen machine generalization).
     - 5-Fold StratifiedGroupKFold grouped by operating condition (Secondary: in-distribution check).
  5. Evaluates Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, Confusion Matrices,
     Classification Reports, and Feature Importances with zero data leakage.
  6. Scientifically selects the best model (ranking by LODO MCC, F1, and Recall), fits it on
     all experiments, and saves artifacts to `models/` and `data/`.

Usage examples:
  python train_pipeline.py --experiment d1_d15_d19
  python train_pipeline.py --experiment d1_d2_d3
  python train_pipeline.py --time-dir "C:/path/to/h5" --boundary-dir "C:/path/to/boundaries" --datasets 1 15 19
  python train_pipeline.py --csv data/processed_features.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from src.feature_extraction import (
    METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    evaluate_physics_stability,
    load_stability_boundary,
    read_experiment_hdf5,
)
from src.preprocessing import (
    build_candidate_models,
    split_features_and_metadata,
    standardize_dataframe_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BOUNDARY_OUT_DIR = DATA_DIR / "stability_boundaries"
SAMPLES_OUT_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"

# Default Windows paths directly matching JOURNAL.ipynb Cell 408
PRESETS = {
    "d1_d15_d19": {
        "name": "D1 + D15 + D19 (Cross-Configuration Domain Shift)",
        "datasets": [1, 15, 19],
        "time_dir": r"C:\Users\Nithanth Devadas\Downloads\intern databse\dataset 1,15,19",
        "boundary_dir": r"C:\Users\Nithanth Devadas\Downloads\intern databse\stability boundary of 1,15, 19",
    },
    "d1_d2_d3": {
        "name": "D1 + D2 + D3 (Initial 394-Experiment Benchmark)",
        "datasets": [1, 2, 3],
        "time_dir": r"C:\Users\Nithanth Devadas\Downloads\intern databse\h1,h2,h3",
        "boundary_dir": r"C:\Users\Nithanth Devadas\Downloads\intern databse\stability boundary of 1,2,3",
    },
}


def find_time_files(time_dir: str, dataset_num: int) -> List[str]:
    """Recursively locate all time{dataset_num}_*.h5 files under time_dir."""
    pattern = re.compile(rf"^time{dataset_num}_(\d+)\.h5$", re.IGNORECASE)
    found: List[Tuple[int, str]] = []
    for root, _, files in os.walk(time_dir):
        for fn in files:
            m = pattern.match(fn)
            if m:
                found.append((int(m.group(1)), os.path.join(root, fn)))
    found.sort(key=lambda pair: pair[0])
    return [path for _, path in found]


def resolve_boundary_path(boundary_dir: str, time_dir: str, dataset_num: int) -> str:
    """Locate stability_boundary{dataset_num}.h5 across boundary_dir, time_dir, or bundled dir."""
    fname = f"stability_boundary{dataset_num}.h5"
    candidates = [
        os.path.join(boundary_dir, fname),
        os.path.join(time_dir, fname),
        str(BOUNDARY_OUT_DIR / fname),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not locate `{fname}`. Checked:\n  - " + "\n  - ".join(candidates)
    )


def extract_dataset_from_hdf5(
    time_dir: str,
    boundary_dir: str,
    datasets: List[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run corrected feature extraction & physics labeling across all requested datasets."""
    if not os.path.isdir(time_dir):
        raise FileNotFoundError(
            f"Time-series folder not found: `{time_dir}`.\n"
            "Pass your exact folder with `--time-dir` and `--boundary-dir`."
        )

    BOUNDARY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    saved_sample_stable = False
    saved_sample_chatter = False
    sampling_rates: List[float] = []

    for ds in datasets:
        b_path = resolve_boundary_path(boundary_dir, time_dir, ds)
        omega_b, b_crit_b = load_stability_boundary(b_path)

        dest_b = BOUNDARY_OUT_DIR / f"stability_boundary{ds}.h5"
        if os.path.abspath(b_path) != os.path.abspath(dest_b):
            shutil.copy2(b_path, dest_b)

        time_files = find_time_files(time_dir, ds)
        if not time_files:
            raise FileNotFoundError(
                f"No `time{ds}_*.h5` files found inside `{time_dir}`."
            )
        print(f"[Dataset {ds}] Extracting {len(time_files)} experiments using {os.path.basename(b_path)}...")

        for fp in time_files:
            parsed = read_experiment_hdf5(fp)
            phys = evaluate_physics_stability(parsed["Omega"], parsed["Depth"], omega_b, b_crit_b)
            chatter = phys["physics_label"]
            sampling_rates.append(parsed["fs"])

            row = {
                "Dataset": ds,
                "Experiment": parsed["Experiment"],
                "File": parsed["File"],
                "Omega": parsed["Omega"],
                "Depth": parsed["Depth"],
                "Critical Depth": phys["critical_depth"],
                "Chatter": chatter,
                "RMS": parsed["RMS"],
                "Peak-to-Peak": parsed["Peak-to-Peak"],
                "Max FFT": parsed["Max FFT"],
                "Dominant Frequency": parsed["Dominant Frequency"],
                "Max Acc": parsed["Max Acc"],
                "Min Acc": parsed["Min Acc"],
                "Sampling Freq (Hz)": parsed["fs"],
            }
            for ch_k, ch_v in parsed["channel_rms"].items():
                row[f"{ch_k}_RMS"] = ch_v
            rows.append(row)

            # Save 1 real Stable & 1 real Chatter sample waveform CSV for the deployed web app
            if chatter == 0 and not saved_sample_stable and ds == datasets[0]:
                sample_df = pd.DataFrame({"time": parsed["time"], "xtacc": parsed["raw_signal"]})
                sample_df.to_csv(SAMPLES_OUT_DIR / "sample_stable_cut.csv", index=False)
                with open(SAMPLES_OUT_DIR / "sample_stable_meta.json", "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "File": parsed["File"],
                            "Dataset": ds,
                            "Omega": parsed["Omega"],
                            "Depth": parsed["Depth"],
                            "Critical Depth": phys["critical_depth"],
                            "True Label": "STABLE",
                        },
                        mf,
                        indent=2,
                    )
                saved_sample_stable = True
            elif chatter == 1 and not saved_sample_chatter and ds == datasets[0]:
                sample_df = pd.DataFrame({"time": parsed["time"], "xtacc": parsed["raw_signal"]})
                sample_df.to_csv(SAMPLES_OUT_DIR / "sample_chatter_cut.csv", index=False)
                with open(SAMPLES_OUT_DIR / "sample_chatter_meta.json", "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "File": parsed["File"],
                            "Dataset": ds,
                            "Omega": parsed["Omega"],
                            "Depth": parsed["Depth"],
                            "Critical Depth": phys["critical_depth"],
                            "True Label": "CHATTER",
                        },
                        mf,
                        indent=2,
                    )
                saved_sample_chatter = True

    df = pd.DataFrame(rows).sort_values(["Dataset", "Experiment"]).reset_index(drop=True)
    extra_info = {
        "sampling_freq_min_hz": float(np.min(sampling_rates)),
        "sampling_freq_max_hz": float(np.max(sampling_rates)),
    }
    return df, extra_info


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Confusion Matrix": cm.tolist(),
        "Classification Report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Stable (0)", "Chatter (1)"],
            output_dict=True,
            zero_division=0,
        ),
    }


def run_lodo_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """Leave-One-Dataset-Out cross-validation with leakage-free scaling per fold."""
    logo = LeaveOneGroupOut()
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError(f"LODO requires at least 2 distinct datasets, found {n_groups}.")

    fold_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []
    aggregate_cms: Dict[str, List[List[int]]] = {}
    aggregate_reports: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in models.items():
        oof_pred = np.zeros(len(y), dtype=int)
        model_folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            clf = clone(pipeline)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = clf.predict(X.iloc[test_idx])
            oof_pred[test_idx] = y_pred

            train_ds = sorted(groups.iloc[train_idx].unique().tolist())
            test_ds = int(groups.iloc[test_idx].iloc[0])
            m = _compute_metrics(y.iloc[test_idx].to_numpy(), y_pred)

            rec = {
                "Model": model_name,
                "Fold": fold_idx,
                "Train Datasets": train_ds,
                "Test Dataset": test_ds,
                "Test Samples": int(len(test_idx)),
                **{k: m[k] for k in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]},
                "Confusion Matrix": m["Confusion Matrix"],
            }
            fold_records.append(rec)
            model_folds.append(rec)

        agg = _compute_metrics(y.to_numpy(), oof_pred)
        aggregate_cms[model_name] = agg["Confusion Matrix"]
        aggregate_reports[model_name] = agg["Classification Report"]

        summary_records.append(
            {
                "Model": model_name,
                "Accuracy": float(np.mean([r["Accuracy"] for r in model_folds])),
                "Balanced Accuracy": float(np.mean([r["Balanced Accuracy"] for r in model_folds])),
                "Precision": float(np.mean([r["Precision"] for r in model_folds])),
                "Recall": float(np.mean([r["Recall"] for r in model_folds])),
                "F1": float(np.mean([r["F1"] for r in model_folds])),
                "MCC": float(np.mean([r["MCC"] for r in model_folds])),
                "Pooled_OOF_Accuracy": agg["Accuracy"],
                "Pooled_OOF_F1": agg["F1"],
                "Pooled_OOF_MCC": agg["MCC"],
            }
        )

    return {
        "folds": fold_records,
        "summary": summary_records,
        "aggregate_confusion_matrices": aggregate_cms,
        "aggregate_classification_reports": aggregate_reports,
    }


def run_grouped_kfold_evaluation(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Any],
    n_splits: int = 5,
) -> Dict[str, Any]:
    """5-Fold StratifiedGroupKFold grouped by Dataset_Omega_Depth (matches JOURNAL Cell 402)."""
    if {"Omega", "Depth"}.issubset(df.columns):
        op_groups = (
            df["Dataset"].astype(str) + "_" + df["Omega"].astype(str) + "_" + df["Depth"].astype(str)
        )
    else:
        op_groups = pd.Series(np.arange(len(df)))

    actual_splits = min(n_splits, int(op_groups.nunique()))
    cv = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    summary_records: List[Dict[str, Any]] = []
    aggregate_cms: Dict[str, List[List[int]]] = {}
    aggregate_reports: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in models.items():
        oof_pred = np.zeros(len(y), dtype=int)
        fold_metrics = []

        for train_idx, test_idx in cv.split(X, y, op_groups):
            clf = clone(pipeline)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = clf.predict(X.iloc[test_idx])
            oof_pred[test_idx] = y_pred
            fold_metrics.append(_compute_metrics(y.iloc[test_idx].to_numpy(), y_pred))

        agg = _compute_metrics(y.to_numpy(), oof_pred)
        aggregate_cms[model_name] = agg["Confusion Matrix"]
        aggregate_reports[model_name] = agg["Classification Report"]

        summary_records.append(
            {
                "Model": model_name,
                "Accuracy": float(np.mean([r["Accuracy"] for r in fold_metrics])),
                "Balanced Accuracy": float(np.mean([r["Balanced Accuracy"] for r in fold_metrics])),
                "Precision": float(np.mean([r["Precision"] for r in fold_metrics])),
                "Recall": float(np.mean([r["Recall"] for r in fold_metrics])),
                "F1": float(np.mean([r["F1"] for r in fold_metrics])),
                "MCC": float(np.mean([r["MCC"] for r in fold_metrics])),
            }
        )

    return {
        "n_splits": actual_splits,
        "summary": summary_records,
        "aggregate_confusion_matrices": aggregate_cms,
        "aggregate_classification_reports": aggregate_reports,
    }


def select_best_model(lodo_summary: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Select deployment model scientifically using LODO MCC, F1, and Recall (not accuracy alone)."""
    ranked = sorted(
        lodo_summary,
        key=lambda r: (r["MCC"], r["F1"], r["Recall"], r["Balanced Accuracy"]),
        reverse=True,
    )
    winner = ranked[0]
    rationale = (
        f"Selected **{winner['Model']}** based on Leave-One-Dataset-Out (LODO) cross-validation "
        f"prioritizing MCC ({winner['MCC']:.3f}), F1-Score ({winner['F1']:.3f}), and Chatter Recall "
        f"({winner['Recall']:.3f}) rather than raw accuracy alone—minimizing undetected chatter "
        "(false negatives) on unseen machining configurations."
    )
    return winner["Model"], rationale


def extract_feature_importances(
    fitted_models: Dict[str, Any],
    feature_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, pipe in fitted_models.items():
        est = pipe.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            out[name] = {
                col: float(val) for col, val in zip(feature_columns, est.feature_importances_)
            }
        elif hasattr(est, "coef_"):
            abs_coef = np.abs(est.coef_[0])
            norm = abs_coef / (np.sum(abs_coef) + 1e-12)
            out[name] = {col: float(val) for col, val in zip(feature_columns, norm)}
    return out


def train_and_save(
    df: pd.DataFrame,
    experiment_label: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = standardize_dataframe_columns(df)
    X, y, groups = split_features_and_metadata(df, MODEL_FEATURE_COLUMNS)

    csv_out = DATA_DIR / "processed_features.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved processed dataset ({len(df)} rows) -> {csv_out}")

    candidates = build_candidate_models(random_state=42)

    print("\nRunning Leave-One-Dataset-Out (LODO) cross-validation...")
    lodo_res = run_lodo_evaluation(X, y, groups, candidates)

    print("Running 5-Fold StratifiedGroupKFold cross-validation...")
    gkf_res = run_grouped_kfold_evaluation(df, X, y, candidates, n_splits=5)

    best_name, rationale = select_best_model(lodo_res["summary"])
    print(f"\n>>> FINAL SELECTED MODEL: {best_name}")
    print(f"    {rationale}\n")

    fitted_all = {}
    for name, pipe in candidates.items():
        clf = clone(pipe)
        clf.fit(X, y)
        fitted_all[name] = clf

    importances = extract_feature_importances(fitted_all, MODEL_FEATURE_COLUMNS)
    final_model = fitted_all[best_name]
    joblib.dump(final_model, MODELS_DIR / "final_model.joblib")

    class_counts = y.value_counts().to_dict()
    ds_counts = groups.value_counts().sort_index().to_dict()
    class_medians = {
        str(cls): {col: float(df.loc[y == cls, col].median()) for col in MODEL_FEATURE_COLUMNS}
        for cls in [0, 1]
    }
    feature_ranges = {
        col: {
            "min": float(X[col].min()),
            "median": float(X[col].median()),
            "max": float(X[col].max()),
        }
        for col in MODEL_FEATURE_COLUMNS
    }

    best_lodo_row = next(r for r in lodo_res["summary"] if r["Model"] == best_name)
    best_gkf_row = next(r for r in gkf_res["summary"] if r["Model"] == best_name)

    metadata = {
        "project_title": "Machine Learning-Based Chatter Detection in Milling Operations",
        "experiment_name": experiment_label,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name,
        "selection_rationale": rationale,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "excluded_metadata_columns": METADATA_COLUMNS,
        "datasets_used": sorted([int(d) for d in groups.unique().tolist()]),
        "dataset_counts": {str(k): int(v) for k, v in ds_counts.items()},
        "total_experiments": int(len(df)),
        "class_distribution": {
            "stable_0": int(class_counts.get(0, 0)),
            "chatter_1": int(class_counts.get(1, 0)),
        },
        "selected_model_lodo_metrics": best_lodo_row,
        "selected_model_grouped_cv_metrics": best_gkf_row,
        "class_feature_medians": class_medians,
        "feature_ranges": feature_ranges,
        "signal_processing": {
            "channel": "xtacc (Column 11)",
            "detrend": "Mean subtraction (DC offset removal)",
            "window": "Hanning window",
            "fft": "Single-sided rFFT normalized by 2 / sum(window), DC bin zeroed",
            "frequency_axis": "True Hz via np.fft.rfftfreq(N, d=mean(diff(time)))",
            **(extra_info or {}),
        },
        "data_source_disclaimer": (
            "Trained exclusively on simulated milling time-series data from the Digital Machining "
            "Database (Schmitz & Smith, Springer, 2019). Not validated on live shop-floor CNC hardware."
        ),
    }

    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    eval_payload = {
        "experiment_name": experiment_label,
        "datasets_used": metadata["datasets_used"],
        "selected_model": best_name,
        "selection_rationale": rationale,
        "lodo": lodo_res,
        "grouped_kfold": gkf_res,
        "feature_importances": importances,
    }
    with open(MODELS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    print("=== LODO CROSS-VALIDATION SUMMARY (Unseen Configuration Generalization) ===")
    print(
        pd.DataFrame(lodo_res["summary"])[
            ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== 5-FOLD STRATIFIED GROUP K-FOLD SUMMARY ===")
    print(
        pd.DataFrame(gkf_res["summary"])[
            ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\nArtifacts saved to:\n"
        f"  - {MODELS_DIR / 'final_model.joblib'}\n"
        f"  - {MODELS_DIR / 'model_metadata.json'}\n"
        f"  - {MODELS_DIR / 'evaluation_results.json'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & validate the Milling Chatter Detection ML pipeline.")
    parser.add_argument(
        "--experiment",
        choices=list(PRESETS.keys()),
        default="d1_d15_d19",
        help="Preset dataset configuration matching your local HDF5 folders (default: d1_d15_d19).",
    )
    parser.add_argument("--time-dir", type=str, default=None, help="Override folder containing time*.h5 files.")
    parser.add_argument("--boundary-dir", type=str, default=None, help="Override folder containing stability_boundary*.h5 files.")
    parser.add_argument("--datasets", type=int, nargs="+", default=None, help="Dataset IDs to include, e.g. --datasets 1 15 19.")
    parser.add_argument("--csv", type=str, default=None, help="Optional: train directly from an already-extracted CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.experiment]

    if args.csv:
        print(f"Loading pre-extracted CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        train_and_save(df, experiment_label=f"CSV Import ({os.path.basename(args.csv)})")
        return

    time_dir = args.time_dir or preset["time_dir"]
    boundary_dir = args.boundary_dir or preset["boundary_dir"]
    datasets = args.datasets or preset["datasets"]
    exp_label = preset["name"] if args.datasets is None else f"Custom Datasets {datasets}"

    df, extra_info = extract_dataset_from_hdf5(time_dir, boundary_dir, datasets)
    train_and_save(df, experiment_label=exp_label, extra_info=extra_info)


if __name__ == "__main__":
    main()
