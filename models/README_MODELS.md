# Models Directory

Run `python train_pipeline.py` from the project root to generate:
- `final_model.joblib`: Fitted scikit-learn Pipeline (with StandardScaler inside pipeline to prevent leakage) for the selected best model.
- `model_metadata.json`: Feature names, metadata columns, dataset composition, training date, and model selection rationale.
- `evaluation_results.json`: Full LODO and 5-Fold StratifiedGroupKFold metrics (Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC), confusion matrices, per-dataset scores, classification reports, and feature importances.
