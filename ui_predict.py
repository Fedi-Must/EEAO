from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


def render_predict_tab(
    df: pd.DataFrame,
    default_feature_value: Callable[[pd.DataFrame, str], Any],
) -> None:
    """Render single-row prediction controls with editable feature selection."""
    st.subheader("Predict one row")

    if st.session_state.tk.model_pipeline is None:
        st.warning("Train or load a model first.")
        return
    if st.session_state.target is None:
        st.warning("Pick a target in the Model tab.")
        return

    tgt = st.session_state.target
    feats = [c for c in df.columns if c != tgt]
    if not feats:
        st.warning("No feature columns are available for prediction.")
        return

    active = [f for f in st.session_state.predict_active_fields if f in feats]
    if not active:
        active = feats[:2] if len(feats) >= 2 else feats[:1]
        st.session_state.predict_active_fields = active

    active = st.multiselect(
        "Editable feature inputs",
        options=feats,
        key="predict_active_fields",
        help="Pick which feature columns you want to edit manually.",
    )
    if not active:
        st.info("Select at least one feature to edit.")

    st.caption(
        "Only selected fields are manually edited. All other features use safe defaults from your dataset."
    )

    inp = {col: default_feature_value(df, col) for col in feats}
    input_cols = st.columns(3)
    for i, col in enumerate(active):
        with input_cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[col]):
                inp[col] = st.number_input(str(col), value=float(inp[col]), key=f"pred_num_{col}")
            else:
                values = df[col].dropna().astype(str).unique().tolist()
                if values:
                    default_val = str(inp[col]) if str(inp[col]) in values else values[0]
                    inp[col] = st.selectbox(
                        str(col),
                        values,
                        index=values.index(default_val),
                        key=f"pred_cat_{col}",
                    )
                else:
                    inp[col] = st.text_input(str(col), value=str(inp[col]), key=f"pred_txt_{col}")

    with st.expander("Current prediction row", expanded=False):
        st.dataframe(pd.DataFrame([inp]), use_container_width=True)

    if st.button("Predict", key="predict_single_row_btn"):
        try:
            pred = st.session_state.tk.predict(inp)
            st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Predict error: {e}")
