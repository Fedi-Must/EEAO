from __future__ import annotations

import time
from typing import Callable

import pandas as pd
import streamlit as st


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
        params = {}
        if "Random Forest" in algo:
            params["n_estimators"] = st.slider("Trees", 10, 300, 100)
        if "KNN" in algo:
            params["n_neighbors"] = st.slider("K", 1, 30, 5)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

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
