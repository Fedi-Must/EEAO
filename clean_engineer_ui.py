from __future__ import annotations

import io
import time
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_ace import st_ace  # type: ignore

    HAS_ACE = True
except Exception:
    HAS_ACE = False


CLEAN_SCRIPT_TEMPLATE = textwrap.dedent(
    """\
    # df is a copy of the current dataset
    # assign your transformed dataframe to new_df
    new_df = df.copy()

    # Example:
    # new_df = new_df.drop_duplicates()
    """
).strip()

EDITOR_THEMES = [
    "tomorrow_night_blue",
    "monokai",
    "cobalt",
    "github",
    "solarized_dark",
]
EDITOR_KEYBINDINGS = ["vscode", "sublime", "emacs", "vim"]

ACTION_SNIPPETS = {
    "Drop duplicates": textwrap.dedent(
        """\
        # Drop duplicate rows
        new_df = df.drop_duplicates().reset_index(drop=True)
        """
    ).strip(),
    "Fill missing (smart)": textwrap.dedent(
        """\
        # Fill missing values per column type
        new_df = df.copy()
        for c in new_df.columns:
            if pd.api.types.is_numeric_dtype(new_df[c]):
                new_df[c] = new_df[c].fillna(new_df[c].median())
            else:
                mode_vals = new_df[c].mode(dropna=True)
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                new_df[c] = new_df[c].fillna(fill_value)
        """
    ).strip(),
    "Drop rows by rule": textwrap.dedent(
        """\
        # Example: drop rows where Age < 18
        new_df = df[df["Age"] >= 18].reset_index(drop=True)
        """
    ).strip(),
    "Drop columns": textwrap.dedent(
        """\
        # Drop specific columns
        cols_to_drop = ["col_a", "col_b"]
        new_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        """
    ).strip(),
    "Rename columns": textwrap.dedent(
        """\
        # Rename columns
        new_df = df.rename(columns={"old_name": "new_name"})
        """
    ).strip(),
    "IQR outlier clip": textwrap.dedent(
        """\
        # Clip outliers for one numeric column using IQR
        new_df = df.copy()
        col = "amount"
        s = new_df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        new_df[col] = s.clip(lo, hi)
        """
    ).strip(),
    "Create feature flag": textwrap.dedent(
        """\
        # Add 0/1 presence flag from a column
        new_df = df.copy()
        source = "email"
        new_df["has_email"] = (
            new_df[source].notna() & new_df[source].astype(str).str.strip().ne("")
        ).astype(int)
        """
    ).strip(),
}

TOOL_ORDER = ["Quick fixes", "Rows", "Columns", "Outliers", "Features", "Script"]
TOOL_ICONS = {
    "Quick fixes": "🛠",
    "Rows": "☰",
    "Columns": "▦",
    "Outliers": "📈",
    "Features": "✨",
    "Script": "</>",
}


def _default_editor_theme() -> str:
    """Return the editor theme that matches the active Streamlit base theme."""
    base = str(st.get_option("theme.base") or "").strip().lower()
    return "monokai" if base == "dark" else "github"


def _init_clean_state() -> None:
    """Initialize session keys used by the Clean + Engineer workspace."""
    state = st.session_state
    if "ce_tool" not in state:
        state.ce_tool = "Quick fixes"
    if "ce_preview_rows" not in state:
        state.ce_preview_rows = 30
    if "ce_script_code" not in state:
        state.ce_script_code = CLEAN_SCRIPT_TEMPLATE
    if "ce_pending_df" not in state:
        state.ce_pending_df = None
    if "ce_script_stdout" not in state:
        state.ce_script_stdout = ""
    if "ce_script_stderr" not in state:
        state.ce_script_stderr = ""
    if "ce_script_traceback" not in state:
        state.ce_script_traceback = ""
    if "ce_editor_theme" not in state:
        state.ce_editor_theme = _default_editor_theme()
    if "ce_editor_keybinding" not in state:
        state.ce_editor_keybinding = "vscode"
    if "ce_editor_font" not in state:
        state.ce_editor_font = 14
    if "ce_editor_wrap" not in state:
        state.ce_editor_wrap = False
    if "ce_editor_height" not in state:
        state.ce_editor_height = 560
    if "ce_canvas_height" not in state:
        state.ce_canvas_height = 520
    if "ce_last_notice" not in state:
        state.ce_last_notice = None


def _render_clean_styles() -> None:
    """Inject CSS styles for the Clean + Engineer shell and cards."""
    st.markdown(
        """
<style>
.ce-shell{
  border:1px solid rgba(28,115,120,.22);
  background:linear-gradient(145deg, rgba(255,255,255,.88), rgba(241,251,252,.92));
  border-radius:18px;
  padding:.75rem 1rem;
  box-shadow:0 10px 28px rgba(10,42,46,.08);
}
.ce-kicker{
  color:#2c7b84;
  text-transform:uppercase;
  letter-spacing:.11em;
  font-size:.72rem;
  margin:0;
}
.ce-title{
  margin:.1rem 0 .25rem 0;
  font-size:1.9rem;
  color:#14232e;
}
.ce-sub{
  margin:0;
  color:#586572;
}
.ce-panel{
  border:1px solid rgba(29,111,115,.22);
  background:linear-gradient(160deg, rgba(255,255,255,.93), rgba(238,249,251,.9));
  border-radius:16px;
  padding:.7rem .85rem .8rem;
}
.ce-panel h3{
  margin:.15rem 0 .2rem 0;
  font-size:1.2rem;
}
.ce-hint{
  color:#4f636f;
  font-size:.92rem;
}
html[data-toolkit-theme="dark"] .ce-shell{
  border:1px solid rgba(96,201,208,.32);
  background:linear-gradient(145deg, rgba(22,32,44,.9), rgba(16,24,34,.94));
  box-shadow:0 12px 28px rgba(2,8,14,.45);
}
html[data-toolkit-theme="dark"] .ce-kicker{color:#8edee5}
html[data-toolkit-theme="dark"] .ce-title{color:#e8f0fa}
html[data-toolkit-theme="dark"] .ce-sub{color:#aab8ca}
html[data-toolkit-theme="dark"] .ce-panel{
  border:1px solid rgba(95,197,203,.32);
  background:linear-gradient(160deg, rgba(23,34,46,.92), rgba(19,28,39,.94));
}
html[data-toolkit-theme="dark"] .ce-hint{color:#a5b4c7}
</style>
""",
        unsafe_allow_html=True,
    )


def _parse_row_index_spec(spec: str, total_rows: int) -> List[int]:
    """Parse row index expressions like '1,4-8' into validated row indices."""
    if not spec.strip():
        return []

    parsed: set[int] = set()
    for token in spec.split(","):
        chunk = token.strip()
        if not chunk:
            continue

        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if idx < 0 or idx >= total_rows:
                    raise ValueError(f"Row index out of range: {idx}")
                parsed.add(idx)
        else:
            idx = int(chunk)
            if idx < 0 or idx >= total_rows:
                raise ValueError(f"Row index out of range: {idx}")
            parsed.add(idx)

    return sorted(parsed)


def _numeric_mask(series: pd.Series, operator: str, value_a: float, value_b: float) -> pd.Series:
    """Build a boolean mask for numeric rule filters."""
    if operator == ">":
        return series > value_a
    if operator == ">=":
        return series >= value_a
    if operator == "<":
        return series < value_a
    if operator == "<=":
        return series <= value_a
    if operator == "==":
        return series == value_a
    if operator == "!=":
        return series != value_a
    return series.between(min(value_a, value_b), max(value_a, value_b), inclusive="both")


def _text_mask(series: pd.Series, operator: str, text_value: str) -> pd.Series:
    """Build a boolean mask for text rule filters."""
    text_series = series.fillna("").astype(str)
    cmp_value = text_value.strip()
    cmp_lower = cmp_value.lower()

    if operator == "contains":
        return text_series.str.contains(cmp_value, case=False, na=False)
    if operator == "equals":
        return text_series.str.lower().eq(cmp_lower)
    if operator == "starts with":
        return text_series.str.lower().str.startswith(cmp_lower)
    if operator == "ends with":
        return text_series.str.lower().str.endswith(cmp_lower)
    if operator == "is empty":
        return text_series.str.strip().eq("")
    return text_series.str.strip().ne("")


def _execute_clean_script(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Execute user script safely and capture stdout, stderr, and traceback."""
    env = {
        "__builtins__": __builtins__,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, env, env)
        trace = ""
    except Exception:
        trace = traceback.format_exc()

    return {
        "ok": trace == "",
        "env": env,
        "stdout": stdout.getvalue().strip(),
        "stderr": stderr.getvalue().strip(),
        "traceback": trace,
    }


def _apply_dataframe_change(
    new_df: pd.DataFrame,
    step_label: str,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
) -> bool:
    """Apply a validated dataframe update and return whether data actually changed."""
    safe_df = validate_df(new_df, step_label)
    current_df = st.session_state.get("data")
    if isinstance(current_df, pd.DataFrame) and safe_df.equals(current_df):
        st.session_state.ce_pending_df = None
        return False

    push_history()
    st.session_state.data = safe_df
    st.session_state.ce_pending_df = None
    log_step(step_label)
    return True


def _insert_snippet(snippet_text: str, mode: str = "append") -> None:
    """Insert a helper snippet into the script buffer (append or replace)."""
    state = st.session_state
    if mode == "replace":
        state.ce_script_code = f"{snippet_text}\n"
    else:
        current = state.ce_script_code.strip()
        if current:
            state.ce_script_code = f"{current}\n\n{snippet_text}\n"
        else:
            state.ce_script_code = f"{snippet_text}\n"


def _run_and_apply(
    action_text: str,
    step_label: str,
    transform_fn: Callable[[], pd.DataFrame],
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Run a transformation callback and apply its dataframe result."""
    try:
        t0 = time.perf_counter()
        with st.spinner(f"{action_text}..."):
            new_df = transform_fn()
        elapsed = time.perf_counter() - t0
        changed = _apply_dataframe_change(
            new_df=new_df,
            step_label=step_label,
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
        )
        if changed:
            st.session_state.ce_last_notice = {
                "level": "success",
                "text": f"{step_label} completed in {elapsed:.2f}s.",
            }
            rerun_app()
        else:
            st.session_state.ce_last_notice = {
                "level": "info",
                "text": f"{step_label}: no changes were needed.",
            }
    except Exception as exc:
        st.session_state.ce_last_notice = {"level": "error", "text": f"{action_text} failed: {exc}"}
        st.error(f"{action_text} failed: {exc}")


def _render_canvas(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
    canvas_height: int,
) -> None:
    """Render main data preview plus pending-transform review controls."""
    state = st.session_state
    st.markdown("### Data Canvas")
    st.caption("Main workspace: preview your data and confirm pending transforms.")

    min_rows = 10
    max_rows = min(max(len(df), 10), 500)
    if state.ce_preview_rows < min_rows:
        state.ce_preview_rows = min_rows
    if state.ce_preview_rows > max_rows:
        state.ce_preview_rows = max_rows

    rows_to_show = st.slider(
        "Rows shown",
        min_value=min_rows,
        max_value=max_rows,
        step=1,
        key="ce_preview_rows",
    )

    preview = df.head(rows_to_show).copy()
    preview.insert(0, "_row", preview.index)
    st.dataframe(preview, use_container_width=True, height=canvas_height)

    pending_df = state.ce_pending_df
    if isinstance(pending_df, pd.DataFrame):
        st.markdown("### Pending Transform Preview")
        st.caption("Review this result before applying it to your dataset.")
        pending_preview = pending_df.head(rows_to_show).copy()
        pending_preview.insert(0, "_row", pending_preview.index)
        st.dataframe(pending_preview, use_container_width=True, height=max(180, int(canvas_height * 0.45)))

        apply_col, discard_col = st.columns(2)
        with apply_col:
            if st.button("Apply pending transform", key="ce_apply_pending"):
                _run_and_apply(
                    action_text="Applying pending transform",
                    step_label="Applied pending transform from script",
                    transform_fn=lambda: pending_df.copy(),
                    validate_df=validate_df,
                    push_history=push_history,
                    log_step=log_step,
                    rerun_app=rerun_app,
                )
        with discard_col:
            if st.button("Discard pending transform", key="ce_discard_pending"):
                state.ce_pending_df = None
                rerun_app()


def _render_quick_fixes(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render one-click cleanup actions such as fill-missing and dedupe."""
    tk = st.session_state.tk

    st.markdown("#### Quick fixes")
    st.caption("One-click cleanup actions with safe undo.")

    if st.button("Undo last step", key="ce_undo"):
        if st.session_state.hist:
            st.session_state.data = st.session_state.hist.pop()
            log_step("Undo")
            st.success("Undone ✅")
            rerun_app()
        else:
            st.info("Nothing to undo.")

    if st.button("Drop duplicates", key="ce_drop_duplicates"):
        deduped_df = df.drop_duplicates().copy()
        removed = len(df) - len(deduped_df)
        _run_and_apply(
            action_text="Removing duplicate rows",
            step_label=f"Drop duplicates: removed {removed}",
            transform_fn=lambda: deduped_df,
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )

    if df.shape[1] == 0:
        st.info("No columns available for missing-value actions.")
        return

    missing_defaults = [col for col in df.columns if df[col].isna().any()][:1]
    if not missing_defaults and len(df.columns) > 0:
        missing_defaults = [df.columns[0]]
    fill_cols = st.multiselect(
        "Columns to fill",
        options=df.columns.tolist(),
        default=missing_defaults,
        key="ce_fill_cols",
    )
    strategy = st.selectbox(
        "Fill strategy",
        ["Auto (Median/Mode)", "Mean", "Median", "Mode", "Zero", "Unknown"],
        key="ce_fill_strategy",
    )
    if st.button("Fill missing values", key="ce_fill_missing"):
        selected_cols = [col for col in fill_cols if col in df.columns]
        if not selected_cols:
            st.warning("Select at least one column to fill.")
        else:
            missing_before = int(df[selected_cols].isna().sum().sum())

            def _bulk_fill_missing() -> pd.DataFrame:
                updated = df.copy()
                for fill_col in selected_cols:
                    updated = tk.fill_missing(updated, fill_col, strategy)
                return updated

            _run_and_apply(
                action_text=f"Filling missing values in {len(selected_cols)} column(s)",
                step_label=(
                    f"Fill missing: {len(selected_cols)} col(s) "
                    f"({strategy}), cells={missing_before}"
                ),
                transform_fn=_bulk_fill_missing,
                validate_df=validate_df,
                push_history=push_history,
                log_step=log_step,
                rerun_app=rerun_app,
            )

    drop_col = st.selectbox("Column for row-drop", df.columns, key="ce_drop_missing_col")

    if st.button("Drop rows with missing in column", key="ce_drop_missing_rows"):
        _run_and_apply(
            action_text=f"Dropping rows with missing {drop_col}",
            step_label=f"Drop rows missing: {drop_col}",
            transform_fn=lambda: tk.drop_missing_rows(df, drop_col),
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )


def _render_rows_tool(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render row-level deletion and rule-based filtering controls."""
    st.markdown("#### Rows")
    st.caption("Drop by row index or by a rule. Row numbers match `_row` in the canvas.")

    row_spec = st.text_input(
        "Drop row indices",
        value="",
        placeholder="Examples: 0, 5, 8-12",
        key="ce_drop_rows_spec",
    )
    if st.button("Drop selected row indices", key="ce_drop_rows_btn"):
        try:
            indexes = _parse_row_index_spec(row_spec, len(df))
            if not indexes:
                st.warning("Enter at least one row index.")
            else:
                _run_and_apply(
                    action_text="Dropping selected rows",
                    step_label=f"Drop rows by index: {indexes[:8]}{'...' if len(indexes) > 8 else ''}",
                    transform_fn=lambda: df.drop(index=indexes).reset_index(drop=True),
                    validate_df=validate_df,
                    push_history=push_history,
                    log_step=log_step,
                    rerun_app=rerun_app,
                )
        except Exception as exc:
            st.error(f"Row drop error: {exc}")

    if df.shape[1] == 0:
        st.info("No columns available for rule-based filtering.")
        return

    st.divider()
    st.caption("Rule-based row filtering")
    rule_column = st.selectbox("Rule column", df.columns, key="ce_rule_col")
    action = st.radio(
        "Action",
        options=["Drop matching rows", "Keep matching rows"],
        horizontal=False,
        key="ce_rule_action",
    )

    series = df[rule_column]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric:
        operator = st.selectbox(
            "Operator",
            [">", ">=", "<", "<=", "==", "!=", "between"],
            key="ce_rule_op_num",
        )
        value_a = st.number_input("Value A", value=0.0, key="ce_rule_num_a")
        value_b = st.number_input("Value B", value=0.0, key="ce_rule_num_b")
        mask = _numeric_mask(series.astype(float), operator, float(value_a), float(value_b))
    else:
        operator = st.selectbox(
            "Operator",
            ["contains", "equals", "starts with", "ends with", "is empty", "is not empty"],
            key="ce_rule_op_txt",
        )
        text_value = "" if operator in {"is empty", "is not empty"} else st.text_input(
            "Text value",
            value="",
            key="ce_rule_txt",
        )
        mask = _text_mask(series, operator, text_value)

    st.caption(f"Matching rows: {int(mask.sum())} / {len(df)}")

    if st.button("Apply row rule", key="ce_apply_rule"):
        if action == "Drop matching rows":
            transform_fn = lambda: df.loc[~mask].copy().reset_index(drop=True)
            action_label = "drop matching"
        else:
            transform_fn = lambda: df.loc[mask].copy().reset_index(drop=True)
            action_label = "keep matching"

        _run_and_apply(
            action_text="Applying row filter rule",
            step_label=f"Row rule ({action_label}): {rule_column} {operator}",
            transform_fn=transform_fn,
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )


def _render_columns_tool(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render column drop and rename tools."""
    tk = st.session_state.tk
    st.markdown("#### Columns")

    if df.shape[1] == 0:
        st.info("No columns available.")
        return

    drop_cols = st.multiselect("Drop columns", df.columns, default=[], key="ce_drop_cols")
    if st.button("Drop selected columns", key="ce_drop_cols_btn"):
        if not drop_cols:
            st.warning("Select columns first.")
        else:
            _run_and_apply(
                action_text="Dropping selected columns",
                step_label=f"Drop columns: {drop_cols}",
                transform_fn=lambda: tk.drop_columns(df, drop_cols),
                validate_df=validate_df,
                push_history=push_history,
                log_step=log_step,
                rerun_app=rerun_app,
            )

    old_name = st.selectbox("Rename column", df.columns, key="ce_rename_old")
    new_name = st.text_input("New name", value=str(old_name), key="ce_rename_new")
    if st.button("Rename column", key="ce_rename_btn"):
        if not new_name.strip():
            st.error("New name cannot be empty.")
        else:
            _run_and_apply(
                action_text="Renaming column",
                step_label=f"Rename: {old_name} -> {new_name.strip()}",
                transform_fn=lambda: tk.rename_column(df, old_name, new_name.strip()),
                validate_df=validate_df,
                push_history=push_history,
                log_step=log_step,
                rerun_app=rerun_app,
            )


def _render_outliers_tool(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render IQR-based outlier preview and clipping controls."""
    tk = st.session_state.tk
    st.markdown("#### Outliers")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
        return

    outlier_col = st.selectbox("Numeric column", num_cols, key="ce_out_col")
    outlier_cols = st.multiselect(
        "Columns to clip",
        options=num_cols,
        default=[outlier_col],
        key="ce_out_cols",
    )
    factor = st.slider("IQR factor", 0.5, 3.0, 1.5, 0.1, key="ce_out_factor")
    series = pd.to_numeric(df[outlier_col], errors="coerce")
    non_null_series = series.dropna()
    if non_null_series.empty:
        st.info("This column has no numeric values to evaluate.")
        return

    q1 = float(non_null_series.quantile(0.25))
    q3 = float(non_null_series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - float(factor) * iqr
    upper = q3 + float(factor) * iqr
    outlier_mask = (series < lower) | (series > upper)
    outlier_count = int(outlier_mask.fillna(False).sum())
    numeric_count = int(non_null_series.shape[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Numeric rows", numeric_count)
    m2.metric("Will clip", outlier_count)
    m3.metric("Will keep", max(0, numeric_count - outlier_count))

    preview_df = pd.DataFrame(
        {
            "row": df.index.astype(int),
            "value": series,
            "status": np.where(outlier_mask.fillna(False), "Will clip", "Keep"),
        }
    ).dropna(subset=["value"])
    if len(preview_df) > 2500:
        preview_df = preview_df.sample(2500, random_state=42).sort_values("row")

    fig = px.scatter(
        preview_df,
        x="row",
        y="value",
        color="status",
        color_discrete_map={"Keep": "#2b8a8e", "Will clip": "#d1495b"},
        title=f"Outlier Preview for {outlier_col}",
        opacity=0.82,
    )
    fig.add_hline(y=lower, line_dash="dash", line_color="#d1495b", annotation_text="Lower clip bound")
    fig.add_hline(y=upper, line_dash="dash", line_color="#d1495b", annotation_text="Upper clip bound")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Blue points stay unchanged. Red points are clipped to the IQR bounds "
        f"[{lower:.3g}, {upper:.3g}] when you apply."
    )

    if st.button("Clip outliers", key="ce_clip_outliers"):
        selected_outlier_cols = [col for col in outlier_cols if col in num_cols]
        if not selected_outlier_cols:
            st.warning("Select at least one numeric column to clip.")
            return

        total_candidates = 0
        for numeric_col in selected_outlier_cols:
            col_series = pd.to_numeric(df[numeric_col], errors="coerce")
            col_clean = col_series.dropna()
            if col_clean.empty:
                continue
            col_q1 = float(col_clean.quantile(0.25))
            col_q3 = float(col_clean.quantile(0.75))
            col_iqr = col_q3 - col_q1
            col_lower = col_q1 - float(factor) * col_iqr
            col_upper = col_q3 + float(factor) * col_iqr
            total_candidates += int(((col_series < col_lower) | (col_series > col_upper)).fillna(False).sum())

        def _bulk_clip_outliers() -> pd.DataFrame:
            updated = df.copy()
            for numeric_col in selected_outlier_cols:
                updated = tk.clip_outliers_iqr(updated, numeric_col, factor=float(factor))
            return updated

        _run_and_apply(
            action_text=f"Clipping outliers in {len(selected_outlier_cols)} column(s)",
            step_label=(
                f"Clip outliers: {len(selected_outlier_cols)} col(s) "
                f"(factor={factor}, candidates={total_candidates})"
            ),
            transform_fn=_bulk_clip_outliers,
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )


def _render_feature_tool(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render lightweight feature engineering helpers."""
    tk = st.session_state.tk
    st.markdown("#### Features")

    if df.shape[1] == 0:
        st.info("No columns available.")
        return

    source_col = st.selectbox("Source column", df.columns, key="ce_feat_src")
    default_name = f"Has{str(source_col).replace(' ', '')}"
    new_col = st.text_input("New 0/1 column", value=default_name, key="ce_feat_new")
    if st.button("Create presence flag", key="ce_feat_btn"):
        _run_and_apply(
            action_text=f"Creating feature {new_col.strip()}",
            step_label=f"Presence flag: {source_col} -> {new_col.strip()}",
            transform_fn=lambda: tk.add_presence_flag(df, source_col, new_col.strip()),
            validate_df=validate_df,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )


def _render_script_tool(df: pd.DataFrame, rerun_app: Callable[[], None]) -> None:
    """Render the script editor, snippet tools, and script execution flow."""
    state = st.session_state
    st.markdown("#### Script IDE")
    st.caption("Larger editor workspace with quick snippets and safer run/apply flow.")

    snippet_col, append_col, replace_col = st.columns([2.4, 1, 1])
    with snippet_col:
        snippet_name = st.selectbox(
            "Snippet library",
            options=list(ACTION_SNIPPETS.keys()),
            key="ce_snippet_pick",
        )
    with append_col:
        if st.button("Append snippet", key="ce_append_snippet"):
            _insert_snippet(ACTION_SNIPPETS[snippet_name], mode="append")
    with replace_col:
        if st.button("Replace with snippet", key="ce_replace_snippet"):
            _insert_snippet(ACTION_SNIPPETS[snippet_name], mode="replace")

    with st.expander("Advanced editor options", expanded=False):
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.selectbox("Theme", EDITOR_THEMES, key="ce_editor_theme")
            st.selectbox("Keymap", EDITOR_KEYBINDINGS, key="ce_editor_keybinding")
        with cfg2:
            st.slider("Font size", 11, 22, key="ce_editor_font")
            st.checkbox("Wrap lines", key="ce_editor_wrap")
            st.slider("Editor height", 420, 980, key="ce_editor_height")

    editor_height = int(state.ce_editor_height)
    state.ce_canvas_height = max(280, min(900, 1100 - int(editor_height * 0.75)))

    if HAS_ACE:
        editor_code = st_ace(
            value=state.ce_script_code,
            language="python",
            theme=state.ce_editor_theme,
            keybinding=state.ce_editor_keybinding,
            font_size=int(state.ce_editor_font),
            tab_size=4,
            show_gutter=True,
            show_print_margin=True,
            wrap=bool(state.ce_editor_wrap),
            min_lines=20,
            height=editor_height,
            auto_update=False,
        )
        if editor_code is not None:
            state.ce_script_code = editor_code
    else:
        st.text_area("Clean script", key="ce_script_code", height=editor_height)
        st.caption("Install richer IDE mode: pip install streamlit-ace")

    run_col, reset_col, clear_col = st.columns([1.2, 1, 1])
    with run_col:
        run_script = st.button("Run script", key="ce_run_script")
    with reset_col:
        if st.button("Reset template", key="ce_reset_script"):
            state.ce_script_code = CLEAN_SCRIPT_TEMPLATE
            state.ce_script_stdout = ""
            state.ce_script_stderr = ""
            state.ce_script_traceback = ""
            state.ce_pending_df = None
    with clear_col:
        if st.button("Clear outputs", key="ce_clear_script_output"):
            state.ce_script_stdout = ""
            state.ce_script_stderr = ""
            state.ce_script_traceback = ""

    if run_script:
        t0 = time.perf_counter()
        with st.spinner("Running custom script..."):
            result = _execute_clean_script(state.ce_script_code, df)
        elapsed = time.perf_counter() - t0
        state.ce_script_stdout = result["stdout"]
        state.ce_script_stderr = result["stderr"]
        state.ce_script_traceback = result["traceback"]

        if not result["ok"]:
            st.error("Script error")
            st.code(state.ce_script_traceback, language="text")
        else:
            env = result["env"]
            new_df = env.get("new_df")
            if isinstance(new_df, pd.DataFrame):
                state.ce_pending_df = new_df
                st.success(f"Script ran in {elapsed:.2f}s. Review pending transform in the canvas.")
            elif "new_df" in env:
                st.warning("`new_df` exists but is not a pandas DataFrame.")
            else:
                st.info(f"Script ran in {elapsed:.2f}s. Set `new_df = ...` to create an apply-ready transform.")

    with st.expander("Script outputs", expanded=bool(state.ce_script_stderr or state.ce_script_traceback)):
        if state.ce_script_stdout:
            st.text_area("stdout", state.ce_script_stdout, height=120, disabled=True)
        if state.ce_script_stderr:
            st.text_area("stderr", state.ce_script_stderr, height=120, disabled=True)


def _set_selected_tool(tool_name: str) -> None:
    """Store the active toolbox section name in session state."""
    st.session_state.ce_tool = tool_name


def _render_tool_selector() -> str:
    """Render tool selector buttons and return the active tool."""
    state = st.session_state
    if state.ce_tool not in TOOL_ORDER:
        state.ce_tool = TOOL_ORDER[0]

    cols = st.columns(len(TOOL_ORDER))
    selected = str(state.ce_tool)
    for idx, tool_name in enumerate(TOOL_ORDER):
        with cols[idx]:
            is_selected = tool_name == selected
            icon = TOOL_ICONS.get(tool_name, "")
            check = " ✓" if is_selected else ""
            label = f"{icon} {tool_name}{check}".strip()
            st.button(
                label,
                key=f"ce_tool_btn_{idx}",
                use_container_width=True,
                on_click=_set_selected_tool,
                args=(tool_name,),
            )
    return str(st.session_state.ce_tool)


def _render_toolbox(
    tool: str,
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Dispatch to the renderer for the selected toolbox module."""
    if tool == "Quick fixes":
        _render_quick_fixes(df, validate_df, push_history, log_step, rerun_app)
    elif tool == "Rows":
        _render_rows_tool(df, validate_df, push_history, log_step, rerun_app)
    elif tool == "Columns":
        _render_columns_tool(df, validate_df, push_history, log_step, rerun_app)
    elif tool == "Outliers":
        _render_outliers_tool(df, validate_df, push_history, log_step, rerun_app)
    elif tool == "Features":
        _render_feature_tool(df, validate_df, push_history, log_step, rerun_app)
    else:
        _render_script_tool(df, rerun_app)


def render_clean_engineer_tab(
    df: pd.DataFrame,
    validate_df: Callable[[pd.DataFrame, str], pd.DataFrame],
    push_history: Callable[[], None],
    log_step: Callable[[str], None],
    rerun_app: Callable[[], None],
) -> None:
    """Render the full Clean + Engineer tab with canvas and toolbox layout."""
    _init_clean_state()
    _render_clean_styles()
    state = st.session_state
    canvas_height = int(max(280, min(900, 1060 - int(state.ce_editor_height))))
    state.ce_canvas_height = canvas_height
    notice = state.ce_last_notice

    st.markdown(
        """
<div class="ce-shell">
  <p class="ce-kicker">Workspace</p>
  <h2 class="ce-title">Clean + Engineer Studio</h2>
  <p class="ce-sub">Futuristic, minimal workflow: canvas in center, tools on the side.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if isinstance(notice, dict):
        message = str(notice.get("text", "")).strip()
        level = str(notice.get("level", "info")).strip().lower()
        if message:
            if level == "success":
                st.success(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)
        state.ce_last_notice = None

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", int(len(df)))
    m2.metric("Columns", int(df.shape[1]))
    m3.metric("Missing", int(df.isnull().sum().sum()))

    st.markdown("### Toolbox")
    st.caption("Pick a workspace tool:")
    tool = _render_tool_selector()

    if tool == "Script":
        canvas_col, tools_col = st.columns([1.4, 3.2])
    else:
        canvas_col, tools_col = st.columns([3.2, 1.4])

    with canvas_col:
        with st.container():
            _render_canvas(df, validate_df, push_history, log_step, rerun_app, canvas_height)
    with tools_col:
        with st.container():
            _render_toolbox(tool, df, validate_df, push_history, log_step, rerun_app)
