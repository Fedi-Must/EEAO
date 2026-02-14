# Data Science Project Toolkit

A Streamlit app for an end-to-end data science workflow: upload data, explore,
clean/engineer, train models, predict, and export a report.

## Requirements
- Python 3.9+
- Git (optional, only for update script)

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## Quick Start (Windows)
- `run_app.bat`
  - Checks Python/pip and shows install guidance if missing.
  - Creates/uses `.venv` automatically.
  - Installs `requirements.txt` automatically when dependencies are missing.
  - Launches `streamlit` app.
- `update_repo.bat`
  - Checks Git and shows install guidance if missing.
  - Runs safe update flow: `git fetch --prune` then `git pull --ff-only`.
  - Explains what to do if local changes/divergence block pull.

## What It Can Do
- Upload CSV or Excel datasets.
- Explore data with quick filters and Plotly charts.
- Clean data (fill missing, drop rows/columns, dedupe).
- Engineer simple features (presence flags, outlier clipping with IQR).
- Compare/train models with baseline metrics.
- Use visual model tuning previews:
  - train/test split preview for `test_size`
  - tournament preview for cross-validation readiness
  - hyperparameter impact charts (Random Forest, KNN, Decision Tree, Logistic)
- Predict one row with editable feature inputs.
- Export a `REPORT.md` summary.
- Optional Gemini assistant in AI Help tab (API key required).
- Local persistence now fully respects `Remember locally`:
  - when disabled, dataset autosave files are not kept/restored
  - autosave restore is blocked until remember mode is enabled

## Update Log (This Release)
Major changes introduced in this modular refactor release:

- Refactored app architecture from monolithic `logic.py` into modular toolkit files:
  - `toolkit_core.py`
  - `toolkit_cleaning.py`
  - `toolkit_modeling.py`
  - `toolkit_reporting.py`
  - `toolkit_visualization.py`
  - unified through `data_toolkit.py`
- Split large tab UIs into focused modules:
  - `ui_model.py`
  - `ui_predict.py`
  - `ui_report.py`
- Introduced dedicated Clean + Engineer workspace module:
  - `clean_engineer_ui.py`
- Added safer dataframe/state guards:
  - no-op change detection (prevents repeated duplicate actions/logs)
  - duplicate consecutive step suppression
- Improved bulk operations in cleaning:
  - multi-column missing-value fill in one click
  - multi-column outlier clipping in one click
- Improved Predict tab input UX:
  - explicit "Editable feature inputs" selector
  - fixes categorical editing issues (for example `Sex`)
- Removed old monolith file:
  - `logic.py`
- Removed optional local smoke script from tracked runtime files:
  - `titanic_manual_smoke.py`

Additional improvements after refactor:
- Modeling UX now includes graphical previews for split behavior and slider impact.
- Model loading now refreshes related metadata/state more safely.
- Stronger upload dedupe signatures reduce accidental reprocessing of same file.
- Added Windows helper scripts:
  - `run_app.bat`
  - `update_repo.bat`

## Future Roadmap
- Add project-level pipeline presets (save/load full cleaning+model workflow).
- Add dataset validation profiles with severity levels and auto-fix suggestions.
- Add model registry view (versioned saved models + metadata history).
- Add richer AI streaming UX (incremental tokens and tool/action traces).
- Add multi-user persistence backend (SQLite/Postgres) instead of only local files.
- Add automated tests (unit + UI smoke) and CI checks on push.

## File Structure And Responsibilities
- `app.py`
  - Main Streamlit shell, session state, persistence hooks, theme bridge, tab wiring.
- `clean_engineer_ui.py`
  - Clean + Engineer tab UI and action handlers.
- `ui_model.py`
  - Model tab UI (task detection, compare/train flows, metrics display, visual tuning previews).
- `ui_predict.py`
  - Predict tab UI (editable input selection + single-row inference).
- `ui_report.py`
  - Report tab UI (generate/download preview).
- `data_toolkit.py`
  - `DataToolkit` class composing all toolkit mixins.
- `toolkit_constants.py`
  - Shared constants (task labels/defaults/thresholds).
- `toolkit_core.py`
  - Shared validation and preprocessing helper utilities.
- `toolkit_cleaning.py`
  - Cleaning and feature-engineering operations.
- `toolkit_modeling.py`
  - Model prep/training/comparison/predict/save/load logic.
- `toolkit_reporting.py`
  - Report and exported training-script generation.
- `toolkit_visualization.py`
  - Suggested visualization helpers.
- `requirements.txt`
  - Python dependencies.
- `run_app.bat`
  - Windows one-click launcher with dependency fail-safes.
- `update_repo.bat`
  - Windows one-click updater (`fetch` + `pull --ff-only`) with guidance.
- `README.md`
  - Project documentation.

## Files Created At Runtime
- `.toolkit_state.json` (saved settings)
- `.toolkit_raw.csv` (original uploaded data)
- `.toolkit_current.csv` (working dataset)
- `model.joblib` (saved model pipeline)
- `.venv/` (created by `run_app.bat` when no virtual environment exists)

## Notes
- Excel support relies on `openpyxl`.
- Markdown tables in reports rely on `tabulate`.
