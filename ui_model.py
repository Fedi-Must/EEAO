from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
import streamlit as st

from toolkit_constants import CLASSIFICATION, DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE


def _safe_split_preview(
    df: pd.DataFrame,
    target: str,
    task: str,
    test_size: float,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], bool, Optional[str]]:
    """Build a deterministic train/test preview using the same split rules as training."""
    features = df.drop(columns=[target])
    labels = df[target]
    if features.shape[1] == 0:
        return None, None, False, "No feature columns found."
    if labels.isna().any():
        return None, None, False, "Target contains missing values."

    stratify_labels: Optional[pd.Series] = None
    stratified = False
    if task == CLASSIFICATION:
        class_counts = labels.value_counts(dropna=False)
        if not class_counts.empty and int(class_counts.min()) >= 2:
            stratify_labels = labels
            stratified = True

    try:
        _, _, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=float(test_size),
            random_state=DEFAULT_RANDOM_STATE,
            stratify=stratify_labels,
        )
    except Exception as exc:
        return None, None, stratified, str(exc)
    return y_train, y_test, stratified, None


def _render_split_preview(target: str, task: str, y_train: pd.Series, y_test: pd.Series, stratified: bool) -> None:
    """Render a graphical preview of what test_size will hold out."""
    st.markdown("#### Split Preview")
    st.caption("Shows how your selected test size partitions rows before training.")

    total_rows = int(len(y_train) + len(y_test))
    p1, p2, p3 = st.columns(3)
    p1.metric("Train rows", int(len(y_train)))
    p2.metric("Test rows", int(len(y_test)))
    p3.metric("Held-out %", f"{(100.0 * len(y_test) / max(1, total_rows)):.1f}%")

    if task == CLASSIFICATION:
        train_counts = y_train.astype(str).value_counts()
        test_counts = y_test.astype(str).value_counts()
        classes = sorted(set(train_counts.index.tolist()) | set(test_counts.index.tolist()))
        preview_rows = []
        for cls in classes:
            preview_rows.append({"Class": cls, "Split": "Train", "Rows": int(train_counts.get(cls, 0))})
            preview_rows.append({"Class": cls, "Split": "Test", "Rows": int(test_counts.get(cls, 0))})
        preview_df = pd.DataFrame(preview_rows)
        fig = px.bar(
            preview_df,
            x="Class",
            y="Rows",
            color="Split",
            barmode="group",
            color_discrete_map={"Train": "#2b8a8e", "Test": "#d1495b"},
            title=f"Class distribution after split ({target})",
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Stratified split is ON."
            if stratified
            else "Stratified split is OFF (class counts are too small for strict stratification)."
        )
    else:
        train_df = pd.DataFrame({"Value": y_train, "Split": "Train"})
        test_df = pd.DataFrame({"Value": y_test, "Split": "Test"})
        preview_df = pd.concat([train_df, test_df], ignore_index=True)
        if len(preview_df) > 6000:
            preview_df = preview_df.sample(6000, random_state=DEFAULT_RANDOM_STATE)
        fig = px.histogram(
            preview_df,
            x="Value",
            color="Split",
            barmode="overlay",
            nbins=35,
            opacity=0.58,
            color_discrete_map={"Train": "#2b8a8e", "Test": "#d1495b"},
            title=f"Target distribution after split ({target})",
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Train/test should look similarly distributed for a stable evaluation.")


def _render_param_impact(algo: str, params: Dict[str, Any], train_rows: int, task: str) -> None:
    """Render visual explanations for active hyperparameters."""
    st.markdown("#### Hyperparameter Impact Preview")

    if "Random Forest" in algo:
        trees = int(params.get("n_estimators", 100))
        min_tree = 10
        tree_axis = np.arange(min_tree, trees + 1, max(1, trees // 18))
        if tree_axis[-1] != trees:
            tree_axis = np.append(tree_axis, trees)
        df_plot = pd.DataFrame({"Trees": tree_axis})
        df_plot["Relative compute"] = df_plot["Trees"] / float(df_plot["Trees"].min())
        df_plot["Ensemble variance (lower is better)"] = (
            1.0 / np.sqrt(df_plot["Trees"])
        ) / float(1.0 / np.sqrt(df_plot["Trees"].min()))
        long_df = df_plot.melt("Trees", var_name="Signal", value_name="Relative value")
        fig = px.line(long_df, x="Trees", y="Relative value", color="Signal", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "More trees increase compute roughly linearly, while prediction stability improves with diminishing returns."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Trees", trees)
        c2.metric("Max depth", "Auto" if params.get("max_depth") is None else int(params["max_depth"]))
        c3.metric("Min leaf samples", int(params.get("min_samples_leaf", 1)))
        return

    if "KNN" in algo:
        neighbors = int(params.get("n_neighbors", 5))
        max_k = max(1, min(train_rows, max(neighbors, 35)))
        k_axis = np.arange(1, max_k + 1, max(1, max_k // 20))
        if k_axis[-1] != neighbors:
            k_axis = np.append(k_axis, neighbors)
        k_axis = np.unique(np.sort(k_axis))
        df_plot = pd.DataFrame({"Neighbors (K)": k_axis})
        df_plot["Relative prediction cost"] = df_plot["Neighbors (K)"] / float(max(1, k_axis.min()))
        df_plot["Smoothing strength"] = df_plot["Neighbors (K)"] / float(max(1, train_rows))
        long_df = df_plot.melt("Neighbors (K)", var_name="Signal", value_name="Relative value")
        fig = px.line(long_df, x="Neighbors (K)", y="Relative value", color="Signal", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher K smooths noise but can miss local patterns. K is also bounded by available training rows."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("K neighbors", neighbors)
        c2.metric("Neighbor share of train set", f"{(100.0 * neighbors / max(1, train_rows)):.1f}%")
        c3.metric("Weighting", str(params.get("weights", "uniform")))
        return

    if "Decision Tree" in algo:
        depth_value = params.get("max_depth")
        depth_for_plot = int(depth_value) if isinstance(depth_value, int) else 22
        depth_axis = np.arange(1, depth_for_plot + 1)
        df_plot = pd.DataFrame({"Depth": depth_axis})
        df_plot["Max possible leaves"] = np.minimum(np.power(2, depth_axis), max(1, train_rows))
        fig = px.line(df_plot, x="Depth", y="Max possible leaves", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Greater depth allows more decision regions; min split/leaf constraints control overfitting.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Max depth", "Auto" if depth_value is None else int(depth_value))
        c2.metric("Min split samples", int(params.get("min_samples_split", 2)))
        c3.metric("Min leaf samples", int(params.get("min_samples_leaf", 1)))
        return

    fit_intercept = bool(params.get("fit_intercept", True))
    if task == CLASSIFICATION:
        c_val = float(params.get("C", 1.0))
        inv_reg = 1.0 / max(1e-6, c_val)
        c_axis = np.linspace(0.05, max(1.0, c_val), 26)
        df_plot = pd.DataFrame({"C": c_axis})
        df_plot["Regularization strength (1/C)"] = 1.0 / df_plot["C"]
        fig = px.line(df_plot, x="C", y="Regularization strength (1/C)", markers=False)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower C applies stronger regularization (simpler boundary), higher C fits training data more closely.")
        c1, c2, c3 = st.columns(3)
        c1.metric("C", f"{c_val:.2f}")
        c2.metric("1/C", f"{inv_reg:.3f}")
        c3.metric("Fit intercept", "Yes" if fit_intercept else "No")
    else:
        st.caption("Linear Regression has fewer tuning knobs; this mostly controls baseline offset behavior.")
        c1, c2 = st.columns(2)
        c1.metric("Fit intercept", "Yes" if fit_intercept else "No")
        c2.metric("Train rows", train_rows)


def _render_tournament_preview(df: pd.DataFrame, target: str, task: str) -> None:
    """Show what model tournament cross-validation will consider."""
    st.markdown("#### Tournament Preview")
    st.caption("This preview shows what cross-validation will evaluate before running all candidates.")
    y = df[target]
    if y.isna().any():
        st.warning("Target has missing values; fill/drop missing target rows before tournament.")
        return

    if task == CLASSIFICATION:
        counts = y.astype(str).value_counts().reset_index()
        counts.columns = ["Class", "Rows"]
        fig = px.bar(counts, x="Class", y="Rows", color="Rows", color_continuous_scale="Teal")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        min_class = int(counts["Rows"].min()) if not counts.empty else 0
        folds = min(DEFAULT_CV_FOLDS, len(df), min_class) if min_class > 0 else 0
        c1, c2 = st.columns(2)
        c1.metric("Smallest class size", min_class)
        c2.metric("Usable CV folds", max(0, folds))
    else:
        preview_df = pd.DataFrame({"Target": y})
        fig = px.histogram(preview_df, x="Target", nbins=35, color_discrete_sequence=["#2b8a8e"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        folds = min(DEFAULT_CV_FOLDS, len(df))
        st.metric("Usable CV folds", max(0, folds))


def render_model_tab(df: pd.DataFrame, log_step: Callable[[str], None]) -> None:
    """Render model selection, comparison, training, and code export controls."""
    st.subheader("Target + task")

    if df.shape[1] == 0:
        st.warning("No columns available for modeling.")
        return

    target = st.selectbox("Target", df.columns, key="target_sel")
    st.session_state.target = target

    task = None
    try:
        task = st.session_state.tk.detect_task_type(df, target)
        st.info(f"Task: {task.upper()}")
    except Exception as e:
        st.error(f"Target error: {e}")

    m1, m2 = st.tabs(["Compare models", "Train one model"])

    with m1:
        if task is not None:
            _render_tournament_preview(df, target, task)
        if st.button("Run tournament"):
            if task is None:
                st.warning("Select a valid target first.")
            else:
                try:
                    with st.spinner("Cross-validation..."):
                        t0 = time.perf_counter()
                        board = st.session_state.tk.compare_models(df, target)
                        dt = time.perf_counter() - t0
                    st.success(f"Done in {dt:.2f}s")
                    st.dataframe(board.style.highlight_max(axis=0), use_container_width=True)
                except Exception as e:
                    st.error(f"AutoML error: {e}")

    with m2:
        algo = st.selectbox("Algorithm", ["Random Forest", "KNN", "Linear/Logistic Regression", "Decision Tree"])
        test_size = st.slider(
            "Test size",
            0.1,
            0.4,
            0.2,
            0.05,
            help="Fraction held out for evaluation. Larger test sets improve evaluation stability but leave less data for training.",
        )

        train_rows = int(max(1, round(len(df) * (1.0 - float(test_size)))))
        if task is not None:
            y_train, y_test, stratified, split_error = _safe_split_preview(df, target, task, float(test_size))
            if split_error:
                st.warning(f"Split preview unavailable: {split_error}")
            elif y_train is not None and y_test is not None:
                train_rows = int(len(y_train))
                _render_split_preview(target, task, y_train, y_test, stratified)

        params: Dict[str, Any] = {}
        if "Random Forest" in algo:
            params["n_estimators"] = st.slider(
                "Trees",
                10,
                400,
                120,
                10,
                key="rf_trees",
                help="More trees usually improve stability but increase training time and memory.",
            )
            rf_auto_depth = st.checkbox(
                "No max depth (grow until stopping rules)",
                value=False,
                key="rf_auto_depth",
            )
            rf_depth = st.slider(
                "Max depth",
                2,
                40,
                12,
                1,
                key="rf_max_depth",
                help="Higher depth captures more complex interactions but can overfit noisy data.",
            )
            params["max_depth"] = None if rf_auto_depth else int(rf_depth)
            params["min_samples_split"] = st.slider(
                "Min samples to split",
                2,
                25,
                2,
                1,
                key="rf_min_samples_split",
                help="A node must have at least this many samples before it can split.",
            )
            params["min_samples_leaf"] = st.slider(
                "Min samples per leaf",
                1,
                25,
                1,
                1,
                key="rf_min_samples_leaf",
                help="Each terminal leaf must keep at least this many samples, which smooths noisy rules.",
            )
        elif "KNN" in algo:
            safe_k_max = max(1, min(80, train_rows))
            default_k = min(5, safe_k_max)
            params["n_neighbors"] = st.slider(
                "K neighbors",
                1,
                safe_k_max,
                default_k,
                1,
                key="knn_neighbors",
                help="Prediction uses the K nearest training rows. Higher K smooths noise but can blur local patterns.",
            )
            params["weights"] = st.selectbox(
                "Neighbor weighting",
                options=["uniform", "distance"],
                key="knn_weights",
                help="`uniform`: all neighbors count equally. `distance`: closer neighbors get more influence.",
            )
        elif "Decision Tree" in algo:
            dt_auto_depth = st.checkbox(
                "No max depth (grow until stopping rules)",
                value=False,
                key="dt_auto_depth",
            )
            dt_depth = st.slider(
                "Max depth",
                2,
                40,
                10,
                1,
                key="dt_max_depth",
                help="Controls the deepest split path. Deeper trees fit more detail.",
            )
            params["max_depth"] = None if dt_auto_depth else int(dt_depth)
            params["min_samples_split"] = st.slider(
                "Min samples to split",
                2,
                25,
                2,
                1,
                key="dt_min_samples_split",
                help="A split is allowed only when a node has at least this many samples.",
            )
            params["min_samples_leaf"] = st.slider(
                "Min samples per leaf",
                1,
                25,
                1,
                1,
                key="dt_min_samples_leaf",
                help="Minimum rows in each terminal leaf. Larger values reduce overfitting.",
            )
        else:
            params["fit_intercept"] = st.checkbox(
                "Fit intercept",
                value=True,
                help="Adds a baseline offset term. Usually keep ON unless your features are already centered.",
            )
            if task == CLASSIFICATION:
                params["C"] = float(
                    st.slider(
                        "Inverse regularization (C)",
                        0.05,
                        10.0,
                        1.0,
                        0.05,
                        help="Lower C = stronger regularization; higher C = more flexible boundary.",
                    )
                )

        _render_param_impact(algo, params, train_rows, task if task is not None else CLASSIFICATION)

        if st.button("Train (with baseline)"):
            if task is None:
                st.warning("Select a valid target first.")
            else:
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
