import json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

from logic import DataToolkit


APP_DIR = Path(".")
STATE_PATH = APP_DIR / ".toolkit_state.json"
RAW_PATH = APP_DIR / ".toolkit_raw.csv"
CUR_PATH = APP_DIR / ".toolkit_current.csv"
MODEL_PATH = APP_DIR / "model.joblib"
MAX_HISTORY = 12


def rr():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def safe_read(f) -> pd.DataFrame:
    n = f.name.lower()
    if n.endswith(".csv"):
        try:
            return pd.read_csv(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pd.read_csv(f, encoding="latin-1")
    return pd.read_excel(f)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(d: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def push_hist():
    if st.session_state.data is None:
        return
    st.session_state.hist.append(st.session_state.data.copy(deep=True))
    if len(st.session_state.hist) > MAX_HISTORY:
        st.session_state.hist = st.session_state.hist[-MAX_HISTORY:]


def log_step(msg: str):
    st.session_state.steps.append(msg)


st.set_page_config(page_title="Data Science Project Toolkit", layout="wide", page_icon="🧪")

if "tk" not in st.session_state:
    st.session_state.tk = DataToolkit()
if "data" not in st.session_state:
    st.session_state.data = None
if "raw" not in st.session_state:
    st.session_state.raw = None
if "base_stats" not in st.session_state:
    st.session_state.base_stats = None
if "uploaded_sig" not in st.session_state:
    st.session_state.uploaded_sig = None
if "hist" not in st.session_state:
    st.session_state.hist = []
if "steps" not in st.session_state:
    st.session_state.steps = []
if "remember" not in st.session_state:
    st.session_state.remember = True
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""
if "x_axis" not in st.session_state:
    st.session_state.x_axis = None
if "y_axis" not in st.session_state:
    st.session_state.y_axis = None
if "plot_type" not in st.session_state:
    st.session_state.plot_type = "Scatter"
if "target" not in st.session_state:
    st.session_state.target = None

if "persist_loaded" not in st.session_state:
    st.session_state.persist_loaded = True
    p = load_state()
    st.session_state.remember = bool(p.get("remember", True))
    st.session_state.gemini_key = str(p.get("gemini_key", "")) if st.session_state.remember else ""
    st.session_state.x_axis = p.get("x_axis") or st.session_state.x_axis
    st.session_state.y_axis = p.get("y_axis") or st.session_state.y_axis
    st.session_state.plot_type = p.get("plot_type") or st.session_state.plot_type
    st.session_state.target = p.get("target") or st.session_state.target


st.markdown(
    """
<style>
.main-header{font-size:2rem;color:#333;text-align:center;margin-bottom:.35rem}
.small-muted{color:#666;font-size:.95rem;text-align:center;margin-bottom:1rem}
.stButton>button{width:100%;border-radius:6px}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<h1 class="main-header">🧪 Data Science Project Toolkit</h1>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Explore ➜ Clean ➜ Engineer ➜ Train ➜ Predict ➜ Report</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("Project Setup")
    st.session_state.remember = st.checkbox("Remember locally", value=st.session_state.remember)
    st.session_state.gemini_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        value=st.session_state.gemini_key if st.session_state.remember else "",
    )

    up = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])

    if up is not None:
        sig = (up.name, getattr(up, "size", None))
        if st.session_state.uploaded_sig != sig:
            try:
                df0 = safe_read(up)
                st.session_state.data = df0
                st.session_state.raw = df0.copy()
                st.session_state.base_stats = stats(df0)
                st.session_state.uploaded_sig = sig
                st.session_state.hist = []
                st.session_state.steps = [f"Loaded dataset: {up.name}"]

                cols = df0.columns.tolist()
                st.session_state.x_axis = cols[0] if cols else None
                st.session_state.y_axis = cols[1] if len(cols) > 1 else (cols[0] if cols else None)

                st.success("Loaded ✅")
            except Exception as e:
                st.error(f"Load error: {e}")

    st.divider()
    st.subheader("Progress")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save"):
            if st.session_state.data is None:
                st.warning("Nothing to save.")
            else:
                try:
                    if st.session_state.raw is not None:
                        st.session_state.raw.to_csv(RAW_PATH, index=False)
                    st.session_state.data.to_csv(CUR_PATH, index=False)
                    save_state({
                        "remember": st.session_state.remember,
                        "gemini_key": st.session_state.gemini_key if st.session_state.remember else "",
                        "x_axis": st.session_state.x_axis,
                        "y_axis": st.session_state.y_axis,
                        "plot_type": st.session_state.plot_type,
                        "target": st.session_state.target,
                    })
                    st.success("Saved ✅")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with c2:
        if st.button("Load"):
            try:
                if CUR_PATH.exists():
                    df1 = pd.read_csv(CUR_PATH)
                    st.session_state.data = df1
                    st.session_state.base_stats = stats(df1)
                    st.session_state.hist = []
                    st.session_state.steps = ["Loaded saved progress"]
                    st.session_state.raw = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else df1.copy()

                    cols = df1.columns.tolist()
                    if cols:
                        if st.session_state.x_axis not in cols:
                            st.session_state.x_axis = cols[0]
                        if st.session_state.y_axis not in cols:
                            st.session_state.y_axis = cols[1] if len(cols) > 1 else cols[0]
                    st.success("Loaded ✅")
                    rr()
                else:
                    st.warning("No saved progress found.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    if st.session_state.raw is not None:
        if st.button("Reset to Upload"):
            st.session_state.data = st.session_state.raw.copy()
            st.session_state.hist = []
            st.session_state.steps = ["Reset to original upload"]
            st.success("Reset ✅")
            rr()

    st.divider()
    st.subheader("Model")
    m1, m2 = st.columns(2)
    with m1:
        if st.button("Save model.joblib"):
            try:
                if st.session_state.tk.model_pipeline is None:
                    st.warning("Train a model first.")
                else:
                    st.session_state.tk.save_model(MODEL_PATH, meta={"target": st.session_state.target})
                    st.success("Saved model.joblib ✅")
            except Exception as e:
                st.error(f"Save model failed: {e}")

    with m2:
        model_up = st.file_uploader("Load .joblib", type=["joblib"], label_visibility="collapsed")
        if model_up is not None:
            try:
                st.session_state.tk.load_model(model_up)
                st.success("Model loaded ✅")
            except Exception as e:
                st.error(f"Load model failed: {e}")

    if st.session_state.remember:
        save_state({
            "remember": True,
            "gemini_key": st.session_state.gemini_key,
            "x_axis": st.session_state.x_axis,
            "y_axis": st.session_state.y_axis,
            "plot_type": st.session_state.plot_type,
            "target": st.session_state.target,
        })
    else:
        save_state({"remember": False})


if st.session_state.data is None:
    st.info("Upload a dataset to begin.")
    st.stop()

df = st.session_state.data
cur = stats(df)
base = st.session_state.base_stats or cur

a, b, c, d = st.columns(4)
a.metric("Rows", cur["rows"], delta=cur["rows"] - base["rows"])
b.metric("Columns", cur["cols"], delta=cur["cols"] - base["cols"])
c.metric("Missing", cur["missing"], delta=cur["missing"] - base["missing"])
d.metric("Duplicates", cur["duplicates"], delta=cur["duplicates"] - base["duplicates"])

t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Clean + Engineer", "Model", "Predict", "Report", "AI Help"])


with t1:
    st.subheader("Preview")

    q = st.text_input("Search columns", value="")
    cols = [x for x in df.columns if q.lower() in str(x).lower()] if q else df.columns.tolist()
    st.caption(f"Columns shown: {len(cols)} / {df.shape[1]}")
    st.dataframe(df[cols].head(25), use_container_width=True)

    st.divider()
    st.subheader("Quick filter")

    fcol = st.selectbox("Filter column", df.columns, key="filter_col")
    if pd.api.types.is_numeric_dtype(df[fcol]):
        mn, mx = float(df[fcol].min()), float(df[fcol].max())
        lo, hi = st.slider("Range", mn, mx, (mn, mx))
        view = df[(df[fcol] >= lo) & (df[fcol] <= hi)]
    else:
        txt = st.text_input("Contains", value="")
        view = df[df[fcol].astype(str).str.contains(txt, case=False, na=False)] if txt else df

    st.dataframe(view.head(25), use_container_width=True)

    st.divider()
    st.subheader("Visualization")

    if st.session_state.x_axis not in df.columns:
        st.session_state.x_axis = df.columns[0]
    if st.session_state.y_axis not in df.columns:
        st.session_state.y_axis = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    s1, _ = st.columns([1, 3])
    with s1:
        if st.button("Swap X ↔ Y"):
            st.session_state.x_axis, st.session_state.y_axis = st.session_state.y_axis, st.session_state.x_axis
            rr()

    x = st.selectbox("X", df.columns, key="x_axis")
    y = st.selectbox("Y", df.columns, key="y_axis")
    k = st.selectbox("Type", ["Scatter", "Box", "Bar", "Histogram"], key="plot_type")

    try:
        if k == "Scatter":
            fig = px.scatter(df, x=x, y=y)
        elif k == "Box":
            fig = px.box(df, x=x, y=y)
        elif k == "Histogram":
            fig = px.histogram(df, x=x)
        else:
            fig = px.bar(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Plot error: {e}")

    st.divider()
    st.subheader("Suggested plots")

    if st.button("Suggest 3 plots"):
        figs = st.session_state.tk.suggest_plots(df, target=st.session_state.target, max_plots=3)
        if not figs:
            st.info("Not enough numeric columns for suggestions.")
        else:
            for f in figs:
                st.plotly_chart(f, use_container_width=True)


with t2:
    st.subheader("Undo")
    u1, _ = st.columns([1, 3])
    with u1:
        if st.button("Undo last step"):
            if st.session_state.hist:
                st.session_state.data = st.session_state.hist.pop()
                log_step("Undo")
                st.success("Undone ✅")
                rr()
            else:
                st.info("Nothing to undo.")

    st.divider()
    st.subheader("Missing + duplicates")

    col = st.selectbox("Column", df.columns, key="clean_col")

    c1, c2, c3 = st.columns(3)
    with c1:
        strat = st.selectbox("Fill strategy", ["Auto (Median/Mode)", "Mean", "Median", "Mode", "Zero", "Unknown"], key="fill_strat")
        if st.button("Apply fill"):
            try:
                push_hist()
                st.session_state.data = st.session_state.tk.fill_missing(df, col, strat)
                log_step(f"Fill missing: {col} ({strat})")
                st.success("Done ✅")
                rr()
            except Exception as e:
                st.error(f"Fill error: {e}")

    with c2:
        if st.button("Drop rows with missing"):
            try:
                push_hist()
                st.session_state.data = st.session_state.tk.drop_missing_rows(df, col)
                log_step(f"Drop rows missing: {col}")
                st.success("Done ✅")
                rr()
            except Exception as e:
                st.error(f"Drop error: {e}")

    with c3:
        if st.button("Drop duplicates"):
            try:
                push_hist()
                before = len(df)
                st.session_state.data = df.drop_duplicates()
                log_step(f"Drop duplicates: removed {before - len(st.session_state.data)}")
                st.success("Done ✅")
                rr()
            except Exception as e:
                st.error(f"Dup error: {e}")

    st.divider()
    st.subheader("Columns")

    dcols = st.multiselect("Drop columns", df.columns, default=[])
    if st.button("Drop selected columns"):
        try:
            if not dcols:
                st.warning("Select columns first.")
            else:
                push_hist()
                st.session_state.data = st.session_state.tk.drop_columns(df, dcols)
                log_step(f"Drop columns: {dcols}")
                st.success("Done ✅")
                rr()
        except Exception as e:
            st.error(f"Drop cols error: {e}")

    r1, r2, r3 = st.columns(3)
    with r1:
        old = st.selectbox("Rename column", df.columns, key="rename_old")
    with r2:
        new = st.text_input("New name", value=str(old), key="rename_new")
    with r3:
        if st.button("Rename"):
            try:
                if not new.strip():
                    st.error("New name cannot be empty.")
                else:
                    push_hist()
                    st.session_state.data = st.session_state.tk.rename_column(df, old, new.strip())
                    log_step(f"Rename: {old} -> {new.strip()}")
                    st.success("Done ✅")
                    rr()
            except Exception as e:
                st.error(f"Rename error: {e}")

    st.divider()
    st.subheader("Outliers (IQR clip)")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
    else:
        oc = st.selectbox("Numeric column", num_cols, key="out_col")
        fac = st.slider("IQR factor", 0.5, 3.0, 1.5, 0.1)
        if st.button("Clip outliers"):
            try:
                push_hist()
                st.session_state.data = st.session_state.tk.clip_outliers_iqr(df, oc, factor=float(fac))
                log_step(f"Clip outliers: {oc} (factor={fac})")
                st.success("Done ✅")
                rr()
            except Exception as e:
                st.error(f"Outlier error: {e}")

    st.divider()
    st.subheader("Feature engineering (presence flag)")

    src = st.selectbox("Source column", df.columns, key="feat_src")
    default = f"Has{str(src).replace(' ', '')}"
    newc = st.text_input("New column", value=default, key="feat_new")
    if st.button("Create 0/1 presence feature"):
        try:
            push_hist()
            st.session_state.data = st.session_state.tk.add_presence_flag(df, src, newc.strip())
            log_step(f"Presence flag: {src} -> {newc.strip()}")
            st.success("Done ✅")
            rr()
        except Exception as e:
            st.error(f"Feature error: {e}")

    st.divider()
    st.subheader("Current preview")
    st.dataframe(st.session_state.data.head(25), use_container_width=True)


with t3:
    st.subheader("AutoML + training")

    target = st.selectbox("Target", df.columns, key="target_sel")
    st.session_state.target = target

    try:
        task = st.session_state.tk.detect_task_type(df, target)
        st.info(f"Task: {task.upper()}")
    except Exception as e:
        st.error(f"Target error: {e}")
        st.stop()

    mode = st.radio("Mode", ["Compare models", "Train one model"], horizontal=True)

    if mode == "Compare models":
        if st.button("Run tournament"):
            try:
                with st.spinner("Cross-validation..."):
                    t0 = time.perf_counter()
                    board = st.session_state.tk.compare_models(df, target)
                    dt = time.perf_counter() - t0
                st.success(f"Done in {dt:.2f}s")
                st.dataframe(board.style.highlight_max(axis=0), use_container_width=True)
            except Exception as e:
                st.error(f"AutoML error: {e}")

    else:
        algo = st.selectbox("Algorithm", ["Random Forest", "KNN", "Linear/Logistic Regression", "Decision Tree"])
        params = {}
        if "Random Forest" in algo:
            params["n_estimators"] = st.slider("Trees", 10, 300, 100)
        if "KNN" in algo:
            params["n_neighbors"] = st.slider("K", 1, 30, 5)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

        if st.button("Train (with baseline)"):
            try:
                with st.spinner("Training..."):
                    t0 = time.perf_counter()
                    result = st.session_state.tk.train_with_baseline(df, target, algo, params, task, float(test_size))
                    dt = time.perf_counter() - t0

                st.success(f"Done in {dt:.2f}s")
                st.write("### Baseline metrics")
                st.json(result["baseline_metrics"])
                st.write("### Model metrics")
                st.json(result["model_metrics"])

                if st.session_state.tk.feature_importance is not None:
                    st.write("### Feature importance (top 20)")
                    st.dataframe(st.session_state.tk.feature_importance.head(20), use_container_width=True)

                st.write("### Export training code")
                st.code(st.session_state.tk.export_code(df, target, algo, params), language="python")

                log_step(f"Train: {algo} (target={target}, test_size={test_size})")
            except Exception as e:
                st.error(f"Train error: {e}")


with t4:
    st.subheader("Predict one row")

    if st.session_state.tk.model_pipeline is None:
        st.warning("Train or load a model first.")
    else:
        if st.session_state.target is None:
            st.warning("Pick a target in the Model tab.")
        else:
            tgt = st.session_state.target
            feats = [c for c in df.columns if c != tgt]
            inp = {}

            cols = st.columns(3)
            for i, col in enumerate(feats):
                with cols[i % 3]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        dv = float(df[col].dropna().mean()) if df[col].dropna().size else 0.0
                        inp[col] = st.number_input(str(col), value=dv)
                    else:
                        vals = df[col].dropna().astype(str).unique().tolist()
                        inp[col] = st.selectbox(str(col), vals) if vals else st.text_input(str(col), value="")

            if st.button("Predict"):
                try:
                    pred = st.session_state.tk.predict(inp)
                    st.success(f"Prediction: {pred}")
                except Exception as e:
                    st.error(f"Predict error: {e}")


with t5:
    st.subheader("Report")
    st.caption("Markdown report: summary, missing table, steps, and metrics.")

    if st.button("Generate report"):
        try:
            rep = st.session_state.tk.build_report(
                df=st.session_state.data,
                raw_df=st.session_state.raw,
                steps=st.session_state.steps,
                target=st.session_state.target,
            )
            st.download_button("Download REPORT.md", data=rep.encode("utf-8"), file_name="REPORT.md", mime="text/markdown")
            st.text_area("Preview", rep, height=350)
        except Exception as e:
            st.error(f"Report error: {e}")


with t6:
    st.subheader("AI Help (Optional)")
    st.caption("Model: gemini-2.5-flash")

    key = st.session_state.gemini_key
    if not HAS_GEMINI:
        st.info("Install (optional): pip install google-generativeai")
    elif not key:
        st.info("Add your API key in the sidebar.")
    else:
        try:
            genai.configure(api_key=key)
            q = st.text_input("Ask about your data / results:")
            if q:
                ctx = (
                    f"Columns: {list(df.columns)}\n"
                    f"Shape: {df.shape}\n\n"
                    f"Head:\n{df.head(8).to_string(index=False)}\n\n"
                    "Explain simply for a school data science project."
                )
                if st.session_state.tk.model_pipeline is not None:
                    ctx += "\nA model exists; interpret metrics and give next steps."
                m = genai.GenerativeModel("gemini-2.5-flash")
                r = m.generate_content(f"{ctx}\n\nUser: {q}")
                st.write(r.text)
        except Exception as e:
            st.error(f"Gemini error: {e}")
