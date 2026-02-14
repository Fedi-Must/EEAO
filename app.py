import io
import json
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px

try:
    import google.generativeai as genai  # type: ignore
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

try:
    from streamlit_ace import st_ace  # type: ignore
    HAS_ACE = True
except Exception:
    HAS_ACE = False

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

from data_toolkit import DataToolkit
from clean_engineer_ui import render_clean_engineer_tab
from ui_model import render_model_tab
from ui_predict import render_predict_tab
from ui_report import render_report_tab


APP_DIR = Path(".")
STATE_PATH = APP_DIR / ".toolkit_state.json"
RAW_PATH = APP_DIR / ".toolkit_raw.csv"
CUR_PATH = APP_DIR / ".toolkit_current.csv"
MODEL_PATH = APP_DIR / "model.joblib"
MAX_HISTORY = 12
PLOT_TYPES = ["Scatter", "Box", "Bar", "Histogram"]
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_OPTIONS = [
    DEFAULT_GEMINI_MODEL,
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-pro",
]
MAX_FIRST_CHAT_CSV_CHARS = 120000


CODE_TEMPLATES = {
    "Quick summary": textwrap.dedent(
        """\
        # df is the current dataset (pandas DataFrame)
        # Set result = ... to display it below
        result = df.describe(include="all")
        """
    ).strip(),
    "Value counts (first column)": textwrap.dedent(
        """\
        col = df.columns[0]
        result = df[col].astype(str).value_counts().head(10)
        """
    ).strip(),
    "Correlation heatmap (numeric)": textwrap.dedent(
        """\
        num = df.select_dtypes(include="number")
        if num.shape[1] >= 2:
            fig = px.imshow(num.corr(numeric_only=True), text_auto=True)
        else:
            result = "Need at least 2 numeric columns."
        """
    ).strip(),
    "Transform + replace dataset": textwrap.dedent(
        """\
        # Create a new DataFrame and assign it to new_df to apply
        new_df = df.copy()
        # Example: drop duplicates
        new_df = new_df.drop_duplicates()
        """
    ).strip(),
}
DEFAULT_CODE = CODE_TEMPLATES["Quick summary"]

def session_defaults() -> Dict[str, Any]:
    """Return the baseline session_state keys and their default values."""
    return {
        "tk": DataToolkit(),
        "data": None,
        "raw": None,
        "base_stats": None,
        "uploaded_sig": None,
        "hist": [],
        "steps": [],
        "remember": True,
        "gemini_key": "",
        "x_axis": None,
        "y_axis": None,
        "plot_type": "Scatter",
        "target": None,
        "code_template": "Quick summary",
        "code_snippet": DEFAULT_CODE,
        "predict_active_fields": [],
        "autosave_enabled": True,
        "dataset_upload_nonce": 0,
        "model_upload_nonce": 0,
        "ai_model_name": DEFAULT_GEMINI_MODEL,
        "ai_chat_messages": [],
        "ai_chat_history": [],
        "ai_dataset_context_sent": False,
        "ai_image_upload_nonce": 0,
        "ai_last_model_name": DEFAULT_GEMINI_MODEL,
    }


def rerun_app() -> None:
    """Trigger a rerun using whichever Streamlit API is available."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def current_streamlit_theme_base() -> str:
    """Resolve the current Streamlit base theme to 'dark' or 'light'."""
    base = str(st.get_option("theme.base") or "").strip().lower()
    return "dark" if base == "dark" else "light"


def default_code_editor_theme() -> str:
    """Select the code editor theme that matches the active Streamlit theme."""
    return "monokai" if current_streamlit_theme_base() == "dark" else "github"


def reset_ai_chat_state(increment_upload_nonce: bool = True) -> None:
    """Clear AI chat state, and optionally refresh the image uploader widget key."""
    st.session_state.ai_chat_messages = []
    st.session_state.ai_chat_history = []
    st.session_state.ai_dataset_context_sent = False
    if increment_upload_nonce:
        st.session_state.ai_image_upload_nonce = int(st.session_state.get("ai_image_upload_nonce", 0)) + 1


def build_first_chat_dataset_context(df: pd.DataFrame) -> str:
    """Build the first-message dataset context, truncating CSV content when needed."""
    csv_payload = df.to_csv(index=False)
    was_truncated = False
    if len(csv_payload) > MAX_FIRST_CHAT_CSV_CHARS:
        csv_payload = csv_payload[:MAX_FIRST_CHAT_CSV_CHARS]
        was_truncated = True

    model_note = "A trained model is available." if st.session_state.tk.model_pipeline is not None else "No model is trained yet."
    truncate_note = "\n[CSV payload truncated for token safety.]" if was_truncated else ""
    return (
        "Dataset context (attach only once at chat start):\n"
        f"- Columns: {list(df.columns)}\n"
        f"- Shape: {df.shape}\n"
        f"- {model_note}\n\n"
        f"CSV:\n{csv_payload}{truncate_note}\n\n"
        "Use this as persistent context for follow-up questions."
    )


def gemini_response_text(response: Any) -> str:
    """Extract a plain-text assistant response from Gemini SDK response objects."""
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", [])
            collected = [str(getattr(part, "text", "")).strip() for part in parts if getattr(part, "text", None)]
            merged = "\n".join([chunk for chunk in collected if chunk])
            if merged:
                return merged
    except Exception:
        pass
    return "No textual response returned."


def stats(df: pd.DataFrame) -> Dict[str, int]:
    """Compute the overview metrics shown in the dashboard stat cards."""
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def validate_candidate_dataframe(candidate: Any, action_text: str) -> pd.DataFrame:
    """Validate transformation output before it replaces the working dataset."""
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError(f"{action_text}: result is not a pandas DataFrame.")

    safe_df = candidate.copy()
    if safe_df.columns.duplicated().any():
        dupes = safe_df.columns[safe_df.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"{action_text}: duplicate column names are not allowed ({dupes[:5]}).")

    safe_df = safe_df.reset_index(drop=True)
    if safe_df.shape[0] == 0:
        raise ValueError(f"{action_text}: operation would leave 0 rows.")
    if safe_df.shape[1] == 0:
        raise ValueError(f"{action_text}: operation would remove all columns.")
    return safe_df


def sync_column_dependent_state(df_now: pd.DataFrame) -> None:
    """Keep all column-driven selections valid after schema changes."""
    columns = df_now.columns.tolist()
    ensure_axis_values(columns)

    if st.session_state.target not in columns:
        st.session_state.target = columns[0] if columns else None

    if "target_sel" in st.session_state and st.session_state.target_sel not in columns:
        st.session_state.target_sel = st.session_state.target


def apply_dataset_change(new_df: pd.DataFrame, step_message: str) -> None:
    """Apply a dataframe update with undo snapshot, logging, and persistence."""
    safe_df = validate_candidate_dataframe(new_df, step_message)
    current_df = st.session_state.data
    if isinstance(current_df, pd.DataFrame) and safe_df.equals(current_df):
        return

    push_history()
    st.session_state.data = safe_df
    if "ce_pending_df" in st.session_state:
        st.session_state.ce_pending_df = None
    sync_column_dependent_state(safe_df)
    log_step(step_message)
    persist_progress()


def undo_last_change(step_message: str = "Undo") -> bool:
    """Restore the latest dataframe snapshot from undo history."""
    if not st.session_state.hist:
        return False

    previous_df = st.session_state.hist.pop()
    safe_df = validate_candidate_dataframe(previous_df, "Undo snapshot")
    st.session_state.data = safe_df
    if "ce_pending_df" in st.session_state:
        st.session_state.ce_pending_df = None
    sync_column_dependent_state(safe_df)
    log_step(step_message)
    persist_progress()
    return True


def ensure_data_state_is_safe() -> bool:
    """Repair or reset invalid in-memory dataframe state and report errors."""
    data = st.session_state.data
    if data is None:
        return True

    try:
        safe_df = validate_candidate_dataframe(data, "Current dataset state")
    except Exception as exc:
        if st.session_state.hist:
            recovered = False
            try:
                recovered = undo_last_change("Auto-recover: reverted invalid state")
            except Exception:
                recovered = False
            if recovered:
                st.warning(f"Recovered from an invalid dataset state: {exc}")
                rerun_app()
                return False

        raw_df = st.session_state.raw
        if isinstance(raw_df, pd.DataFrame):
            try:
                safe_raw = validate_candidate_dataframe(raw_df, "Raw recovery")
                st.session_state.data = safe_raw
                st.session_state.hist = []
                sync_column_dependent_state(safe_raw)
                log_step("Auto-recover: restored uploaded dataset")
                st.warning(f"Recovered from an invalid dataset state: {exc}")
                rerun_app()
                return False
            except Exception:
                pass

        st.error(f"Dataset state is invalid and could not be recovered: {exc}")
        return False

    st.session_state.data = safe_df
    sync_column_dependent_state(safe_df)
    return True


def render_keyboard_shortcuts() -> None:
    """Inject browser-side keyboard shortcuts for save, undo, and reset actions."""
    components.html(
        """
<script>
(function () {
  const parentDoc = window.parent && window.parent.document;
  if (!parentDoc || parentDoc.__toolkitUndoBound) return;

  parentDoc.__toolkitUndoBound = true;
  parentDoc.addEventListener("keydown", function (event) {
    const key = (event.key || "").toLowerCase();
    const isUndo = (event.ctrlKey || event.metaKey) && key === "z";
    if (!isUndo) return;

    const t = event.target;
    const tag = (t && t.tagName ? t.tagName : "").toLowerCase();
    const editing = tag === "input" || tag === "textarea" || (t && t.isContentEditable);
    if (editing) return;

    const buttons = Array.from(parentDoc.querySelectorAll("button"));
    const undo = buttons.find((b) => (b.innerText || "").trim() === "Undo (Ctrl+Z)");
    if (!undo) return;

    event.preventDefault();
    undo.click();
  });
})();
</script>
""",
        height=0,
    )


def render_drag_drop_overlay() -> None:
    """Inject browser-side drag/drop listeners to style file-drop feedback."""
    components.html(
        """
<script>
(function () {
  const doc = window.parent && window.parent.document;
  if (!doc || doc.__toolkitDragBound) return;
  doc.__toolkitDragBound = true;

  let depth = 0;
  const hasFiles = (evt) =>
    evt && evt.dataTransfer && Array.from(evt.dataTransfer.types || []).includes("Files");

  doc.addEventListener("dragenter", (evt) => {
    if (!hasFiles(evt)) return;
    depth += 1;
    doc.body.classList.add("toolkit-drag-active");
  });

  doc.addEventListener("dragover", (evt) => {
    if (!hasFiles(evt)) return;
    evt.preventDefault();
    doc.body.classList.add("toolkit-drag-active");
  });

  doc.addEventListener("dragleave", (evt) => {
    if (!hasFiles(evt)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) {
      doc.body.classList.remove("toolkit-drag-active");
    }
  });

  doc.addEventListener("drop", (evt) => {
    if (!hasFiles(evt)) return;
    depth = 0;
    doc.body.classList.remove("toolkit-drag-active");
  });
})();
</script>
""",
        height=0,
    )


def render_theme_bridge() -> None:
    """Mirror Streamlit runtime theme onto CSS data attributes used by custom UI."""
    components.html(
        """
<script>
(function () {
  const doc = window.parent && window.parent.document;
  if (!doc) return;

  const parseColor = (value) => {
    if (!value) return null;
    const v = String(value).trim();
    const rgbMatch = v.match(/^rgba?\\((\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?),\\s*(\\d+(?:\\.\\d+)?)/i);
    if (rgbMatch) {
      return [
        Math.max(0, Math.min(255, Number(rgbMatch[1]))),
        Math.max(0, Math.min(255, Number(rgbMatch[2]))),
        Math.max(0, Math.min(255, Number(rgbMatch[3]))),
      ];
    }
    const hexMatch = v.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hexMatch) {
      const h = hexMatch[1];
      if (h.length === 3) {
        return [
          parseInt(h[0] + h[0], 16),
          parseInt(h[1] + h[1], 16),
          parseInt(h[2] + h[2], 16),
        ];
      }
      return [
        parseInt(h.slice(0, 2), 16),
        parseInt(h.slice(2, 4), 16),
        parseInt(h.slice(4, 6), 16),
      ];
    }
    return null;
  };

  const luminance = (rgb) => {
    const [r, g, b] = rgb.map((x) => x / 255);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const normalizeTheme = (value) => {
    const raw = String(value || "").toLowerCase();
    if (raw.includes("dark")) return "dark";
    if (raw.includes("light")) return "light";
    return null;
  };

  const inferTheme = () => {
    const attrCandidates = [
      doc.documentElement.getAttribute("data-theme"),
      doc.body && doc.body.getAttribute("data-theme"),
      doc.documentElement.dataset && doc.documentElement.dataset.theme,
      doc.body && doc.body.dataset && doc.body.dataset.theme,
      doc.documentElement.className,
      doc.body && doc.body.className,
    ];
    for (const candidate of attrCandidates) {
      const normalized = normalizeTheme(candidate);
      if (normalized) return normalized;
    }

    const rootStyle = getComputedStyle(doc.documentElement);
    const streamlitBg = parseColor(rootStyle.getPropertyValue("--background-color"));
    if (streamlitBg) {
      return luminance(streamlitBg) < 0.5 ? "dark" : "light";
    }
    const streamlitText = parseColor(rootStyle.getPropertyValue("--text-color"));
    if (streamlitText) {
      return luminance(streamlitText) > 0.6 ? "dark" : "light";
    }

    if (window.matchMedia) {
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return "light";
  };

  const updateTheme = () => {
    const nextTheme = inferTheme();
    const root = doc.documentElement;
    if (root.getAttribute("data-toolkit-theme") !== nextTheme) {
      root.setAttribute("data-toolkit-theme", nextTheme);
    }
  };

  updateTheme();

  if (!doc.__toolkitThemeInterval) {
    doc.__toolkitThemeInterval = window.setInterval(updateTheme, 350);
  }

  window.addEventListener("focus", updateTheme, { passive: true });
  doc.addEventListener("visibilitychange", updateTheme, { passive: true });
})();
</script>
""",
        height=0,
    )


def persist_progress() -> None:
    """Persist current workspace data and metadata when remember mode is enabled."""
    state = st.session_state
    if state.data is None:
        return
    try:
        if state.raw is not None:
            state.raw.to_csv(RAW_PATH, index=False)
        state.data.to_csv(CUR_PATH, index=False)
        save_state(state_payload())
    except Exception:
        pass


def handle_uploaded_dataset(uploaded_file: Any, source_label: str = "upload") -> None:
    """Load an uploaded dataset and reset dependent state for a fresh workflow."""
    if uploaded_file is None:
        return

    signature = (uploaded_file.name, getattr(uploaded_file, "size", None))
    if st.session_state.uploaded_sig == signature:
        return

    uploaded_df = safe_read(uploaded_file)
    set_dataset_state(
        df_new=uploaded_df,
        raw_df=uploaded_df.copy(),
        steps=[f"Loaded dataset ({source_label}): {uploaded_file.name}"],
        reset_axes=True,
    )
    st.session_state.uploaded_sig = signature
    persist_progress()


def restore_progress_if_available() -> bool:
    """Attempt to restore remembered workspace progress on startup."""
    if st.session_state.data is not None:
        return True
    try:
        if load_saved_progress():
            return True
    except Exception:
        return False
    return False


def safe_read(file_obj: Any) -> pd.DataFrame:
    """Read CSV/XLSX uploads with parsing fallbacks and strict validation."""
    file_name = file_obj.name.lower()
    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(file_obj)
        except UnicodeDecodeError:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding="latin-1")
    return pd.read_excel(file_obj)


def load_state() -> Dict[str, Any]:
    """Load persisted non-dataset preferences from disk, if available."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(payload: Dict[str, Any]) -> None:
    """Persist non-dataset preferences to disk."""
    try:
        STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def push_history() -> None:
    """Push the current dataframe onto undo history."""
    if st.session_state.data is None:
        return
    st.session_state.hist.append(st.session_state.data.copy(deep=True))
    if len(st.session_state.hist) > MAX_HISTORY:
        st.session_state.hist = st.session_state.hist[-MAX_HISTORY:]


def log_step(message: str) -> None:
    """Append an entry to the user-visible processing step log."""
    text = str(message).strip()
    if not text:
        return
    steps = st.session_state.steps
    if steps and str(steps[-1]).strip() == text:
        return
    steps.append(text)


def init_session_state() -> None:
    """Initialize session_state defaults and normalize missing/legacy keys."""
    state = st.session_state
    for key, value in session_defaults().items():
        if key not in state:
            state[key] = value.copy() if isinstance(value, list) else value

    if "persist_loaded" not in state:
        state.persist_loaded = True
        persisted = load_state()
        state.remember = bool(persisted.get("remember", True))
        state.gemini_key = str(persisted.get("gemini_key", "")) if state.remember else ""
        state.x_axis = persisted.get("x_axis") or state.x_axis
        state.y_axis = persisted.get("y_axis") or state.y_axis
        state.plot_type = persisted.get("plot_type") or state.plot_type
        state.target = persisted.get("target") or state.target
        persisted_predict_fields = persisted.get("predict_active_fields")
        if isinstance(persisted_predict_fields, list):
            state.predict_active_fields = persisted_predict_fields
        persisted_tool = persisted.get("ce_tool")
        if isinstance(persisted_tool, str) and persisted_tool.strip():
            state.ce_tool = persisted_tool
        persisted_ai_model = persisted.get("ai_model_name")
        if isinstance(persisted_ai_model, str) and persisted_ai_model in GEMINI_MODEL_OPTIONS:
            state.ai_model_name = persisted_ai_model
        state.ai_last_model_name = state.ai_model_name


def state_payload() -> Dict[str, Any]:
    """Create the serializable preferences payload for local persistence."""
    return {
        "remember": st.session_state.remember,
        "gemini_key": st.session_state.gemini_key if st.session_state.remember else "",
        "x_axis": st.session_state.x_axis,
        "y_axis": st.session_state.y_axis,
        "plot_type": st.session_state.plot_type,
        "target": st.session_state.target,
        "predict_active_fields": st.session_state.predict_active_fields,
        "ce_tool": st.session_state.get("ce_tool", "Quick fixes"),
        "ai_model_name": st.session_state.get("ai_model_name", DEFAULT_GEMINI_MODEL),
    }


def ensure_axis_values(columns: List[str]) -> None:
    """Ensure plot axis selections stay valid after column changes."""
    if not columns:
        st.session_state.x_axis = None
        st.session_state.y_axis = None
        return
    if st.session_state.x_axis not in columns:
        st.session_state.x_axis = columns[0]
    if st.session_state.y_axis not in columns:
        st.session_state.y_axis = columns[1] if len(columns) > 1 else columns[0]


def default_feature_value(df: pd.DataFrame, col: str) -> Any:
    """Compute a safe default input value for manual prediction forms."""
    if pd.api.types.is_numeric_dtype(df[col]):
        clean = df[col].dropna()
        return float(clean.mean()) if clean.size else 0.0
    clean_text = df[col].dropna().astype(str)
    if not clean_text.empty:
        mode_vals = clean_text.mode()
        if not mode_vals.empty:
            return str(mode_vals.iloc[0])
        return str(clean_text.iloc[0])
    return ""


def set_dataset_state(df_new: pd.DataFrame, raw_df: pd.DataFrame, steps: List[str], reset_axes: bool) -> None:
    """Atomically replace dataset state and synchronize related UI controls."""
    safe_df = validate_candidate_dataframe(df_new, "Dataset load")
    safe_raw = validate_candidate_dataframe(raw_df, "Raw dataset load")

    reset_ai_chat_state(increment_upload_nonce=True)
    st.session_state.data = safe_df
    st.session_state.raw = safe_raw
    st.session_state.base_stats = stats(safe_df)
    st.session_state.hist = []
    st.session_state.steps = steps

    columns = safe_df.columns.tolist()
    if reset_axes:
        st.session_state.x_axis = columns[0] if columns else None
        st.session_state.y_axis = columns[1] if len(columns) > 1 else (columns[0] if columns else None)
    else:
        ensure_axis_values(columns)

    if st.session_state.target not in columns:
        st.session_state.target = columns[0] if columns else None
    st.session_state.target_sel = st.session_state.target
    if not st.session_state.predict_active_fields:
        candidates = [c for c in columns if c != st.session_state.target]
        st.session_state.predict_active_fields = candidates[:2]
    persist_progress()


def load_saved_progress() -> bool:
    """Load persisted raw/current datasets and history snapshots from disk."""
    if not CUR_PATH.exists():
        return False

    current_df = pd.read_csv(CUR_PATH)
    raw_df = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else current_df.copy()
    set_dataset_state(
        df_new=current_df,
        raw_df=raw_df,
        steps=["Loaded saved progress"],
        reset_axes=False,
    )
    return True


def clear_saved_dataset_files() -> None:
    """Remove persisted dataset files used by remember-progress mode."""
    for file_path in (RAW_PATH, CUR_PATH):
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


def reset_workspace_state() -> None:
    """Reset workspace data and UI state while preserving selected preferences."""
    remember_pref = bool(st.session_state.get("remember", True))
    gemini_key = str(st.session_state.get("gemini_key", ""))
    ai_model_name = str(st.session_state.get("ai_model_name", DEFAULT_GEMINI_MODEL))
    dataset_upload_nonce = int(st.session_state.get("dataset_upload_nonce", 0)) + 1
    model_upload_nonce = int(st.session_state.get("model_upload_nonce", 0)) + 1

    for key, value in session_defaults().items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value

    for key in list(st.session_state.keys()):
        if key.startswith(("ce_", "pred_num_", "pred_cat_", "pred_txt_")):
            del st.session_state[key]

    st.session_state.remember = remember_pref
    st.session_state.gemini_key = gemini_key if remember_pref else ""
    st.session_state.ai_model_name = (
        ai_model_name if ai_model_name in GEMINI_MODEL_OPTIONS else DEFAULT_GEMINI_MODEL
    )
    st.session_state.ai_last_model_name = st.session_state.ai_model_name
    st.session_state.dataset_upload_nonce = dataset_upload_nonce
    st.session_state.model_upload_nonce = model_upload_nonce
    st.session_state.persist_loaded = True
    st.session_state.uploaded_sig = None

    clear_saved_dataset_files()
    if st.session_state.remember:
        save_state(state_payload())
    else:
        save_state({"remember": False})


def execute_user_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Run user script against a dataframe copy and return execution artifacts."""
    env = {
        "__builtins__": __builtins__,
        "df": df.copy(),
        "raw_df": st.session_state.raw,
        "pd": pd,
        "np": np,
        "px": px,
        "tk": st.session_state.tk,
    }
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, env, env)
    except Exception:
        return {
            "ok": False,
            "traceback": traceback.format_exc(),
            "stdout": stdout.getvalue().strip(),
            "stderr": stderr.getvalue().strip(),
            "env": env,
        }
    return {
        "ok": True,
        "traceback": "",
        "stdout": stdout.getvalue().strip(),
        "stderr": stderr.getvalue().strip(),
        "env": env,
    }


def swap_axes() -> None:
    """Swap X and Y axis selections in the visualization builder."""
    x_val = st.session_state.x_axis
    y_val = st.session_state.y_axis
    st.session_state.x_axis = y_val
    st.session_state.y_axis = x_val


st.set_page_config(page_title="Data Science Project Toolkit", layout="wide")
init_session_state()


render_theme_bridge()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --ink:#1a2430;
  --muted:#5d6976;
  --accent:#0f6f7a;
  --accent-2:#22a6a0;
  --panel:rgba(255,255,255,.88);
  --bg-main:
    radial-gradient(1100px 440px at 10% -15%, rgba(34,166,160,.15), transparent 55%),
    radial-gradient(900px 420px at 100% -10%, rgba(15,111,122,.13), transparent 58%),
    linear-gradient(180deg, #eef6fa 0%, #f7fbfd 100%);
  --side-bg:linear-gradient(180deg, rgba(245,251,253,.96) 0%, rgba(236,247,250,.98) 100%);
  --tab-ink:#435463;
  --tab-active:#0e5d67;
  --drop-bg:rgba(232,246,249,.92);
  --drop-border:rgba(15,111,122,.46);
  --input-bg:rgba(255,255,255,.92);
  --input-ink:#1a2430;
  --input-border:rgba(15,111,122,.28);
  --input-focus:#1f8c95;
  --chat-surface:rgba(255,255,255,.85);
}
html[data-toolkit-theme="dark"]{
  --ink:#e6eef7;
  --muted:#a7b6c8;
  --accent:#66d0d9;
  --accent-2:#4fbdb6;
  --panel:rgba(20,29,38,.82);
  --bg-main:
    radial-gradient(1100px 440px at 10% -15%, rgba(79,189,182,.18), transparent 58%),
    radial-gradient(900px 420px at 100% -10%, rgba(40,110,160,.18), transparent 62%),
    linear-gradient(180deg, #0f161f 0%, #111a24 100%);
  --side-bg:linear-gradient(180deg, rgba(17,25,34,.98) 0%, rgba(14,20,30,.98) 100%);
  --tab-ink:#b5c3d6;
  --tab-active:#8de2dc;
  --drop-bg:rgba(12,22,32,.92);
  --drop-border:rgba(102,208,217,.55);
  --input-bg:rgba(18,28,39,.9);
  --input-ink:#e6eef7;
  --input-border:rgba(102,208,217,.32);
  --input-focus:#66d0d9;
  --chat-surface:rgba(20,29,38,.78);
}
html, body, [data-testid="stAppViewContainer"]{
  background:var(--bg-main);
  color:var(--ink);
  font-family:"Space Grotesk", sans-serif;
}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{
  background:var(--side-bg);
  border-right:1px solid rgba(15,111,122,.16);
}
[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"]{
  color:var(--muted) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
div[data-baseweb="base-input"] > div,
div[data-baseweb="select"] > div{
  background:var(--input-bg) !important;
  color:var(--input-ink) !important;
  border:1px solid var(--input-border) !important;
  border-radius:10px !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder{
  color:var(--muted) !important;
}
div[data-baseweb="select"] *{
  color:var(--input-ink) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
  border-color:var(--input-focus) !important;
  box-shadow:0 0 0 1px var(--input-focus) inset !important;
}
[data-testid="stChatMessage"]{
  border:1px solid rgba(15,111,122,.2);
  border-radius:14px;
  background:var(--chat-surface);
  padding:.2rem .35rem;
}
.main-header{
  font-size:2.15rem;
  color:var(--ink);
  text-align:center;
  margin-bottom:.25rem;
  letter-spacing:.01em;
}
.small-muted{
  color:var(--muted);
  font-size:.98rem;
  text-align:center;
  margin-bottom:.8rem
}
.stButton>button{
  width:100%;
  border-radius:10px;
  border:1px solid rgba(15,111,122,.26);
  background:linear-gradient(160deg, rgba(18,122,132,.93), rgba(11,94,103,.96));
  color:#f3fcfd;
  font-weight:600;
  transition:transform .14s ease, box-shadow .14s ease, filter .14s ease;
}
.stButton>button:hover{
  transform:translateY(-1px);
  filter:brightness(1.03);
  box-shadow:0 8px 20px rgba(13,81,88,.22);
}
[data-testid="stMetric"]{
  background:var(--panel);
  border:1px solid rgba(16,108,118,.2);
  border-radius:14px;
  padding:.4rem .55rem;
}
div[data-baseweb="tab-list"]{
  gap:.25rem;
  border-bottom:1px solid rgba(13,105,114,.16);
}
div[data-baseweb="tab-list"] button{
  border-radius:10px 10px 0 0;
  color:var(--tab-ink);
  padding:.42rem .8rem;
}
div[data-baseweb="tab-list"] button[aria-selected="true"]{
  color:var(--tab-active);
  background:linear-gradient(180deg, rgba(34,166,160,.14), rgba(34,166,160,.03));
  border-bottom:2px solid #179590;
  font-weight:600;
}
[data-testid="stDataFrame"]{
  border:1px solid rgba(14,101,109,.16);
  border-radius:12px;
}
code, pre, .stCodeBlock, textarea{
  font-family:"JetBrains Mono", monospace !important;
}

#toolkit-drop-anchor + div{
  position:fixed;
  inset:0;
  z-index:1000;
  pointer-events:none;
  opacity:0;
  transition:opacity .2s ease;
}
#toolkit-drop-anchor + div [data-testid="stFileUploaderDropzone"]{
  height:100vh;
  min-height:100vh;
  border:2px dashed var(--drop-border);
  border-radius:0;
  background:var(--drop-bg);
  color:var(--ink);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);
}
#toolkit-drop-anchor + div section{
  height:100vh;
}
#toolkit-drop-anchor + div [data-testid="stFileUploaderDropzone"] div{
  font-size:1.2rem;
}
body.toolkit-drag-active #toolkit-drop-anchor + div{
  opacity:1;
  pointer-events:auto;
}
body.toolkit-drag-active #toolkit-drop-anchor + div [data-testid="stFileUploaderDropzone"]{
  animation:dropPulse .95s ease-in-out infinite alternate;
}
@keyframes dropPulse{
  from{box-shadow:inset 0 0 0 1px rgba(255,255,255,.05), 0 0 0 0 rgba(34,166,160,.25);}
  to{box-shadow:inset 0 0 0 1px rgba(255,255,255,.05), 0 0 0 20px rgba(34,166,160,0);}
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<h1 class="main-header">Data Science Project Toolkit</h1>', unsafe_allow_html=True)
st.markdown('<div class="small-muted">Explore ➜ Clean ➜ Engineer ➜ Train ➜ Predict ➜ Report</div>', unsafe_allow_html=True)
render_keyboard_shortcuts()
render_drag_drop_overlay()

show_top_dropzone = st.session_state.data is None and not CUR_PATH.exists()
if show_top_dropzone:
    st.markdown('<div id="toolkit-drop-anchor"></div>', unsafe_allow_html=True)
    global_drop_upload = st.file_uploader(
        "Drop CSV/Excel anywhere to upload",
        type=["csv", "xlsx"],
        key=f"global_drop_upload_{st.session_state.dataset_upload_nonce}",
        label_visibility="collapsed",
    )
    if global_drop_upload is not None:
        try:
            handle_uploaded_dataset(global_drop_upload, source_label="dropzone")
            st.success(f"Loaded from dropzone: {global_drop_upload.name}")
        except Exception as exc:
            st.error(f"Drop upload error: {exc}")


with st.sidebar:
    state = st.session_state
    st.header("Project Setup")
    with st.expander("Account & keys", expanded=True):
        state.remember = st.checkbox("Remember locally", value=state.remember)
        state.gemini_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            value=state.gemini_key if state.remember else "",
        )

    with st.expander("Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload Dataset (CSV/Excel)",
            type=["csv", "xlsx"],
            key=f"sidebar_upload_{state.dataset_upload_nonce}",
        )
        if uploaded_file is not None:
            try:
                handle_uploaded_dataset(uploaded_file, source_label="sidebar")
                st.success("Loaded ✅")
            except Exception as exc:
                st.error(f"Load error: {exc}")
        st.caption("Tip: You can also drag a file anywhere on the screen.")

    with st.expander("Persistence", expanded=False):
        st.caption("Progress auto-saves after every change.")
        if st.button("Force Save Now"):
            persist_progress()
            st.success("Saved ✅")
        if st.button("Restore Last Autosave"):
            try:
                if load_saved_progress():
                    st.success("Restored ✅")
                else:
                    st.warning("No autosave found.")
            except Exception as exc:
                st.error(f"Restore failed: {exc}")

        if state.raw is not None:
            if st.button("Reset to Upload"):
                try:
                    apply_dataset_change(state.raw.copy(), "Reset to original upload")
                except Exception as exc:
                    st.error(f"Reset failed: {exc}")
                else:
                    st.success("Reset ✅")
        st.caption("Need a fully clean workspace for a new dataset?")
        if st.button("Reset Workspace (New Dataset)"):
            reset_workspace_state()
            rerun_app()
    with st.expander("Model files", expanded=False):
        st.caption("This section is optional. Use only if you need to export/import trained models.")
        if st.button("Save model.joblib"):
            try:
                if state.tk.model_pipeline is None:
                    st.warning("Train a model first.")
                else:
                    state.tk.save_model(MODEL_PATH, meta={"target": state.target})
                    st.success("Saved model.joblib ✅")
            except Exception as exc:
                st.error(f"Save model failed: {exc}")
        uploaded_model = st.file_uploader(
            "Load model.joblib",
            type=["joblib"],
            key=f"model_upload_{state.model_upload_nonce}",
        )
        if uploaded_model is not None:
            try:
                state.tk.load_model(uploaded_model)
                st.success("Model loaded ✅")
            except Exception as exc:
                st.error(f"Load model failed: {exc}")

    with st.expander("Steps log", expanded=False):
        if state.steps:
            st.markdown("\n".join([f"- {item}" for item in state.steps[-10:]]))
        else:
            st.caption("No steps yet.")

    with st.expander("Quick actions", expanded=True):
        st.caption("Use undo if a transform gave an unexpected result.")
        if st.button("Undo (Ctrl+Z)", key="global_undo_btn"):
            try:
                if undo_last_change("Undo"):
                    st.success("Undone ✅")
                else:
                    st.info("Nothing to undo.")
            except Exception as exc:
                st.error(f"Undo failed: {exc}")

    if state.remember:
        persist_progress()
    else:
        save_state({"remember": False})


if st.session_state.data is None:
    restored = restore_progress_if_available()
    if not restored:
        st.info("Upload a dataset to begin.")
        st.stop()

if not ensure_data_state_is_safe():
    st.stop()

df = st.session_state.data
cur = stats(df)
base = st.session_state.base_stats or cur

a, b, c, d = st.columns(4)
a.metric("Rows", cur["rows"], delta=cur["rows"] - base["rows"])
b.metric("Columns", cur["cols"], delta=cur["cols"] - base["cols"])
c.metric("Missing", cur["missing"], delta=cur["missing"] - base["missing"])
d.metric("Duplicates", cur["duplicates"], delta=cur["duplicates"] - base["duplicates"])

t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["📊 Overview", "🧼 Clean + Engineer", "🧠 Model", "🔮 Predict", "📝 Report", "💻 Code Lab", "🤖 AI Help"]
)


with t1:
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Data explorer")
        if df.shape[1] == 0:
            st.warning("No columns available. Undo the last action to continue.")
        else:
            with st.expander("Preview", expanded=True):
                q = st.text_input("Search columns", value="")
                cols = [x for x in df.columns if q.lower() in str(x).lower()] if q else df.columns.tolist()
                st.caption(f"Columns shown: {len(cols)} / {df.shape[1]}")
                st.dataframe(df[cols].head(25), use_container_width=True)

            with st.expander("Quick filter", expanded=False):
                fcol = st.selectbox("Filter column", df.columns, key="filter_col")
                if pd.api.types.is_numeric_dtype(df[fcol]):
                    mn = float(df[fcol].min()) if df[fcol].notna().any() else 0.0
                    mx = float(df[fcol].max()) if df[fcol].notna().any() else 0.0
                    if mn == mx:
                        lo, hi = mn, mx
                    else:
                        lo, hi = st.slider("Range", mn, mx, (mn, mx))
                    view = df[(df[fcol] >= lo) & (df[fcol] <= hi)]
                else:
                    txt = st.text_input("Contains", value="")
                    view = df[df[fcol].astype(str).str.contains(txt, case=False, na=False)] if txt else df

                st.dataframe(view.head(25), use_container_width=True)

    with right:
        st.subheader("Visualization")
        if df.shape[1] == 0:
            st.info("No columns available for plotting.")
        else:
            with st.expander("Plot builder", expanded=True):
                ensure_axis_values(df.columns.tolist())

                s1, _ = st.columns([1, 3])
                with s1:
                    st.button("Swap X ↔ Y", on_click=swap_axes, key="swap_xy_btn")

                x = st.selectbox("X", df.columns, key="x_axis")
                y = st.selectbox("Y", df.columns, key="y_axis")
                k = st.selectbox("Type", PLOT_TYPES, key="plot_type")

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

            with st.expander("Suggested plots", expanded=False):
                if st.button("Suggest 3 plots"):
                    figs = st.session_state.tk.suggest_plots(df, target=st.session_state.target, max_plots=3)
                    if not figs:
                        st.info("Not enough numeric columns for suggestions.")
                    else:
                        for f in figs:
                            st.plotly_chart(f, use_container_width=True)


with t2:
    try:
        render_clean_engineer_tab(
            df=df,
            validate_df=validate_candidate_dataframe,
            push_history=push_history,
            log_step=log_step,
            rerun_app=rerun_app,
        )
    except Exception as e:
        st.error(f"Clean + Engineer error: {e}")


with t3:
    render_model_tab(df=df, log_step=log_step)


with t4:
    render_predict_tab(df=df, default_feature_value=default_feature_value)


with t5:
    render_report_tab()


with t6:
    st.subheader("Code Lab")
    st.caption("Run custom Python against your current dataset. Use result, fig, or new_df.")

    c1, c2 = st.columns([2, 1])
    with c1:
        template = st.selectbox("Starter templates", list(CODE_TEMPLATES.keys()), key="code_template")
    with c2:
        if st.button("Load template"):
            st.session_state.code_snippet = CODE_TEMPLATES[template]

    if HAS_ACE:
        code = st_ace(
            value=st.session_state.code_snippet,
            language="python",
            theme=default_code_editor_theme(),
            keybinding="vscode",
            font_size=14,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            min_lines=16,
            height=300,
            auto_update=False,
        )
        if code is not None:
            st.session_state.code_snippet = code
    else:
        st.text_area("Python code", key="code_snippet", height=260)
        st.caption("Install the IDE editor: pip install streamlit-ace")

    st.caption("Available objects: df (copy), raw_df, pd, np, px, tk. Set result or fig for output; set new_df to apply changes.")

    run_c1, run_c2, _ = st.columns([1, 1, 3])
    run = False
    with run_c1:
        run = st.button("Run code")
    with run_c2:
        if st.button("Reset code"):
            st.session_state.code_snippet = DEFAULT_CODE
            rerun_app()

    if run:
        code = st.session_state.code_snippet
        if not code.strip():
            st.warning("Write some code to run.")
        else:
            with st.spinner("Running code..."):
                exec_result = execute_user_code(code, df)
            if not exec_result["ok"]:
                st.error("Code error")
                st.code(exec_result["traceback"], language="text")
            else:
                out_text = exec_result["stdout"]
                err_text = exec_result["stderr"]
                env = exec_result["env"]

                if out_text:
                    st.text_area("Output", out_text, height=160)
                if err_text:
                    st.text_area("Errors", err_text, height=160)

                result_obj = env.get("result", env.get("output"))
                if result_obj is not None:
                    if isinstance(result_obj, pd.DataFrame):
                        st.dataframe(result_obj, use_container_width=True)
                    elif isinstance(result_obj, pd.Series):
                        st.dataframe(result_obj.to_frame(), use_container_width=True)
                    else:
                        st.write(result_obj)

                fig_obj = env.get("fig")
                if fig_obj is not None:
                    st.plotly_chart(fig_obj, use_container_width=True)

                new_df = env.get("new_df")
                if isinstance(new_df, pd.DataFrame):
                    st.subheader("new_df preview")
                    st.dataframe(new_df.head(25), use_container_width=True)
                    if st.button("Apply new_df to dataset"):
                        try:
                            apply_dataset_change(new_df, "Custom code: applied new_df")
                        except Exception as exc:
                            st.error(f"Apply failed: {exc}")
                        else:
                            st.success("Applied ✅")
                            rerun_app()
                elif "new_df" in env:
                    st.warning("new_df exists but is not a pandas DataFrame.")

                if not out_text and not err_text and result_obj is None and fig_obj is None and "new_df" not in env:
                    st.info("No output yet. Use print(), set result, or set fig.")


with t7:
    st.subheader("AI Help")
    st.caption("Chat-style Gemini assistant with dataset memory.")

    st.markdown(
        """
<style>
.ai-help-note{
  border:1px solid rgba(15,111,122,.24);
  background:linear-gradient(180deg, rgba(34,166,160,.12), rgba(34,166,160,.04));
  border-radius:12px;
  padding:.5rem .7rem;
  color:var(--ink);
  margin-bottom:.55rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    key = st.session_state.gemini_key
    if not HAS_GEMINI:
        st.info("Install (optional): pip install google-generativeai")
    elif not key:
        st.info("Add your API key in the sidebar.")
    else:
        st.markdown(
            '<div class="ai-help-note">CSV context is attached only on the first message of each chat.</div>',
            unsafe_allow_html=True,
        )
        controls_left, controls_right = st.columns([3, 1])
        with controls_left:
            st.selectbox("Gemini model", options=GEMINI_MODEL_OPTIONS, key="ai_model_name")
        with controls_right:
            st.write("")
            if st.button("New chat", key="ai_new_chat_btn"):
                reset_ai_chat_state(increment_upload_nonce=True)
                st.session_state.ai_last_model_name = st.session_state.ai_model_name
                rerun_app()

        if st.session_state.ai_last_model_name != st.session_state.ai_model_name:
            reset_ai_chat_state(increment_upload_nonce=True)
            st.session_state.ai_last_model_name = st.session_state.ai_model_name

        image_upload = st.file_uploader(
            "Attach image (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"ai_image_upload_{st.session_state.ai_image_upload_nonce}",
        )
        if image_upload is not None:
            st.caption(f"Attached image: {image_upload.name}")

        for message in st.session_state.ai_chat_messages:
            with st.chat_message("assistant" if message.get("role") == "assistant" else "user"):
                st.markdown(str(message.get("content", "")))
                image_bytes = message.get("image_bytes")
                image_name = str(message.get("image_name", "image"))
                if isinstance(image_bytes, (bytes, bytearray)):
                    st.image(image_bytes, caption=image_name, use_column_width=True)

        with st.form("ai_chat_form", clear_on_submit=True):
            prompt = st.text_input(
                "Message Gemini about this dataset...",
                key="ai_prompt_draft",
                placeholder="Ask about patterns, cleaning choices, models, or next steps...",
            )
            send_prompt = st.form_submit_button("Send")

        if send_prompt and not prompt.strip():
            st.warning("Type a message first.")

        if send_prompt and prompt.strip():
            prompt = prompt.strip()
            image_bytes = image_upload.getvalue() if image_upload is not None else None
            image_name = image_upload.name if image_upload is not None else ""
            mime_type = (getattr(image_upload, "type", "") if image_upload is not None else "") or "image/png"

            user_display_message = {"role": "user", "content": prompt}
            if isinstance(image_bytes, (bytes, bytearray)):
                user_display_message["image_bytes"] = image_bytes
                user_display_message["image_name"] = image_name
            st.session_state.ai_chat_messages.append(user_display_message)

            with st.chat_message("user"):
                st.markdown(prompt)
                if isinstance(image_bytes, (bytes, bytearray)):
                    st.image(image_bytes, caption=image_name or "uploaded image", use_column_width=True)

            request_text = prompt
            if not st.session_state.ai_dataset_context_sent:
                request_text = f"{build_first_chat_dataset_context(df)}\n\nUser message:\n{prompt}"

            request_parts: List[Any] = [request_text]
            if isinstance(image_bytes, (bytes, bytearray)):
                if HAS_PIL:
                    request_parts.append(Image.open(io.BytesIO(image_bytes)))
                else:
                    request_parts.append({"mime_type": mime_type, "data": image_bytes})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        genai.configure(api_key=key)
                        model = genai.GenerativeModel(st.session_state.ai_model_name)
                        chat = model.start_chat(history=st.session_state.ai_chat_history)
                        payload: Any = request_parts if len(request_parts) > 1 else request_parts[0]
                        response = chat.send_message(payload)
                        answer = gemini_response_text(response)

                        st.session_state.ai_chat_history.append({"role": "user", "parts": [request_text]})
                        st.session_state.ai_chat_history.append({"role": "model", "parts": [answer]})
                        st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
                        st.session_state.ai_dataset_context_sent = True
                    except Exception as e:
                        answer = f"Gemini error: {e}"
                        st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)

            if isinstance(image_bytes, (bytes, bytearray)):
                st.session_state.ai_image_upload_nonce = int(st.session_state.ai_image_upload_nonce) + 1
