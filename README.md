# Data Science Project Toolkit

A Streamlit app for an end-to-end data science workflow: upload data, explore,
clean/engineer, train models, predict, and export a report.

## Requirements
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## What It Can Do
- Upload CSV or Excel datasets.
- Explore data with quick filters and Plotly charts.
- Clean data (fill missing, drop rows/columns, dedupe).
- Engineer simple features (presence flags, outlier clipping with IQR).
- Compare/train models with baseline metrics.
- Predict one row with editable feature inputs.
- Export a `REPORT.md` summary.
- Optional Gemini assistant in AI Help tab (API key required).

## Update Log (Compared To Current GitHub `main`)
Compared against `origin/main` on February 14, 2026.

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
  - Model tab UI (task detection, compare/train flows, metrics display).
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
- `README.md`
  - Project documentation.

## Files Created At Runtime
- `.toolkit_state.json` (saved settings)
- `.toolkit_raw.csv` (original uploaded data)
- `.toolkit_current.csv` (working dataset)
- `model.joblib` (saved model pipeline)

## Notes
- Excel support relies on `openpyxl`.
- Markdown tables in reports rely on `tabulate`.
