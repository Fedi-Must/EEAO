# Data Science Project Toolkit

A Streamlit app that guides a full, beginner-friendly data science workflow:
upload data, explore, clean, engineer features, train models, predict, and
export a markdown report.

## What it can do
- Upload CSV or Excel datasets.
- Explore data with quick filtering and Plotly charts.
- Clean data (fill missing values, drop rows/columns, remove duplicates).
- Engineer features (presence flags, outlier clipping with IQR).
- Train and compare models with a baseline.
- Predict on a single row using a trained or loaded model.
- Export a REPORT.md summary.
- Optional Gemini AI help for quick explanations (requires API key).

## Requirements
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Optional AI helper
The **AI Help** tab uses Google Gemini. Add your API key in the sidebar to
enable it. If you do not plan to use it, you can ignore the key input.

## Files created
The app stores progress in the project folder:
- `.toolkit_state.json` – saved settings (optional)
- `.toolkit_raw.csv` – original uploaded data
- `.toolkit_current.csv` – current working data
- `model.joblib` – saved model pipeline (when exported)

## Notes
- Excel uploads rely on `openpyxl`.
- Report generation uses `tabulate`.
