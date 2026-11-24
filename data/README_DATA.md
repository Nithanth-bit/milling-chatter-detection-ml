# Data Directory

- `dataset_configurations.csv`: Verified modal and cutting parameters extracted from `Dataset descriptions .xlsx` (Schmitz & Smith Digital Machining Database).
- `journal_documented_results.json`: Verified benchmark results explicitly recorded in `JOURNAL.ipynb`.
- `stability_boundaries/`: Analytical stability boundary HDF5 files (`stability_boundary1.h5`, `stability_boundary2.h5`, `stability_boundary3.h5` included; running `train_pipeline.py` also copies `stability_boundary15.h5` and `stability_boundary19.h5` here).
- `processed_features.csv`: Generated automatically when you run `python train_pipeline.py`.
- `samples/`: Sample Stable and Chatter CSV waveforms exported automatically by `train_pipeline.py` so the deployed web app works out-of-the-box for public visitors.
