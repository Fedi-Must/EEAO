from __future__ import annotations

from typing import Dict, Any, Optional, List
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import joblib


class DataToolkit:
    def __init__(self):
        self.model_pipeline: Optional[Pipeline] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.task_type: Optional[str] = None
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.last_baseline: Optional[Dict[str, Any]] = None

    def _v(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise ValueError("Dataset is empty.")

    def detect_task_type(self, df: pd.DataFrame, target: str) -> str:
        self._v(df)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")
        if df[target].nunique(dropna=True) < 10 or df[target].dtype == "object":
            return "classification"
        return "regression"

    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        num = X.select_dtypes(include=["number"]).columns
        cat = X.select_dtypes(include=["object", "category", "bool"]).columns
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)])
        return ColumnTransformer([("num", num_t, num), ("cat", cat_t, cat)], remainder="drop")

    def compare_models(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        self._v(df)
        task = self.detect_task_type(df, target)
        X, y = df.drop(columns=[target]), df[target]
        if X.shape[1] == 0:
            raise ValueError("No features found.")

        if task == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
            }
            scoring = "accuracy"
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
                "Ridge": Ridge(),
            }
            scoring = "r2"

        pre = self.get_preprocessor(X)
        rows = []
        for name, m in models.items():
            pipe = Pipeline([("preprocessor", pre), ("model", m)])
            s = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
            rows.append({"Model": name, "Score": float(s.mean()), "Std": float(s.std())})

        return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    def _make_model(self, task: str, algo: str, params: Dict[str, Any]):
        if task == "classification":
            if "Random Forest" in algo:
                return RandomForestClassifier(random_state=42, **params)
            if "KNN" in algo:
                return KNeighborsClassifier(**params)
            if "Linear/Logistic" in algo or "Logistic" in algo:
                return LogisticRegression(max_iter=1000, **params)
            return DecisionTreeClassifier(random_state=42, **params)
        else:
            if "Random Forest" in algo:
                return RandomForestRegressor(random_state=42, **params)
            if "KNN" in algo:
                return KNeighborsRegressor(**params)
            if "Linear/Logistic" in algo or "Linear" in algo:
                return LinearRegression(**params)
            return Ridge(**params)

    def _baseline(self, task: str):
        return DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor(strategy="mean")

    def _metrics(self, task: str, y_true, y_pred) -> Dict[str, float]:
        if task == "classification":
            return {"Accuracy": float(accuracy_score(y_true, y_pred)), "F1": float(f1_score(y_true, y_pred, average="weighted"))}
        return {"R2": float(r2_score(y_true, y_pred)), "MSE": float(mean_squared_error(y_true, y_pred))}

    def train_with_baseline(self, df: pd.DataFrame, target: str, algo: str, params: Dict[str, Any], task_type: str, test_size: float) -> Dict[str, Any]:
        self._v(df)
        X, y = df.drop(columns=[target]), df[target]
        if X.shape[1] == 0:
            raise ValueError("No features found.")

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=42)
        pre = self.get_preprocessor(Xtr)

        base = Pipeline([("preprocessor", pre), ("model", self._baseline(task_type))])
        base.fit(Xtr, ytr)
        bmet = self._metrics(task_type, yte, base.predict(Xte))

        model = self._make_model(task_type, algo, params)
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        mmet = self._metrics(task_type, yte, pipe.predict(Xte))

        self.model_pipeline = pipe
        self.task_type = task_type
        self.last_metrics = {"model": mmet, "test_size": float(test_size), "algo": algo, "params": params}
        self.last_baseline = {"baseline": bmet, "strategy": "most_frequent" if task_type == "classification" else "mean"}
        self.feature_importance = self._feature_importance()

        return {"baseline_metrics": bmet, "model_metrics": mmet}

    def _feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model_pipeline is None:
            return None
        pre = self.model_pipeline.named_steps.get("preprocessor")
        m = self.model_pipeline.named_steps.get("model")
        try:
            names = pre.get_feature_names_out()
        except Exception:
            return None

        imp = None
        if hasattr(m, "feature_importances_"):
            imp = np.asarray(m.feature_importances_, dtype=float)
        elif hasattr(m, "coef_"):
            imp = np.abs(np.asarray(m.coef_, dtype=float)).ravel()

        if imp is None:
            return None

        n = min(len(names), len(imp))
        out = pd.DataFrame({"feature": names[:n], "importance": imp[:n]})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    def predict(self, row: Dict[str, Any]) -> Any:
        if self.model_pipeline is None:
            raise ValueError("Model not trained/loaded.")
        return self.model_pipeline.predict(pd.DataFrame([row]))[0]

    def save_model(self, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
        if self.model_pipeline is None:
            raise ValueError("No model to save.")
        payload = {"pipeline": self.model_pipeline, "task_type": self.task_type, "meta": meta or {}, "metrics": self.last_metrics, "baseline": self.last_baseline}
        joblib.dump(payload, path)

    def load_model(self, file_obj) -> None:
        payload = joblib.load(file_obj)
        if isinstance(payload, dict) and "pipeline" in payload:
            self.model_pipeline = payload["pipeline"]
            self.task_type = payload.get("task_type")
            self.last_metrics = payload.get("metrics")
            self.last_baseline = payload.get("baseline")
        else:
            self.model_pipeline = payload
            self.task_type = None

    def fill_missing(self, df: pd.DataFrame, col: str, strat: str) -> pd.DataFrame:
        self._v(df)
        if col not in df.columns:
            raise ValueError("Column not found.")
        out = df.copy()
        s = out[col]
        if strat == "Auto (Median/Mode)":
            strat = "Median" if pd.api.types.is_numeric_dtype(s) else "Mode"

        if strat == "Mean":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError("Mean requires numeric column.")
            out[col] = s.fillna(float(s.mean()))
        elif strat == "Median":
            if not pd.api.types.is_numeric_dtype(s):
                raise ValueError("Median requires numeric column.")
            out[col] = s.fillna(float(s.median()))
        elif strat == "Mode":
            m = s.mode(dropna=True)
            out[col] = s.fillna(m.iloc[0] if not m.empty else None)
        elif strat == "Zero":
            out[col] = s.fillna(0)
        elif strat == "Unknown":
            if pd.api.types.is_numeric_dtype(s):
                raise ValueError("Unknown is for categorical columns.")
            out[col] = s.fillna("Unknown")
        else:
            raise ValueError("Unknown strategy.")
        return out

    def drop_missing_rows(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        self._v(df)
        if col not in df.columns:
            raise ValueError("Column not found.")
        return df.dropna(subset=[col]).copy()

    def drop_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        self._v(df)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"Columns not found: {miss}")
        return df.drop(columns=cols).copy()

    def rename_column(self, df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
        self._v(df)
        if old not in df.columns:
            raise ValueError("Old column not found.")
        if new in df.columns and new != old:
            raise ValueError("New name already exists.")
        return df.rename(columns={old: new}).copy()

    def clip_outliers_iqr(self, df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
        self._v(df)
        if col not in df.columns:
            raise ValueError("Column not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError("Requires numeric column.")
        s = df[col].astype(float)
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        out = df.copy()
        out[col] = s.clip(lo, hi)
        return out

    def add_presence_flag(self, df: pd.DataFrame, source_col: str, new_col: str) -> pd.DataFrame:
        self._v(df)
        if source_col not in df.columns:
            raise ValueError("Source column not found.")
        if not new_col or not str(new_col).strip():
            raise ValueError("New column name cannot be empty.")
        if new_col in df.columns:
            raise ValueError("New column already exists.")
        out = df.copy()
        s = out[source_col]
        present = s.notna()
        if s.dtype == "object" or str(s.dtype).startswith("string"):
            present = present & s.astype(str).str.strip().ne("")
        out[new_col] = present.astype(int)
        return out

    def suggest_plots(self, df: pd.DataFrame, target: Optional[str] = None, max_plots: int = 3):
        self._v(df)
        figs = []
        d = df.copy()

        if target and target in d.columns:
            try:
                figs.append(px.histogram(d, x=target, title="Target distribution"))
            except Exception:
                pass

        num = d.select_dtypes(include=["number"]).columns.tolist()
        if len(num) >= 2:
            try:
                corr = d[num].corr(numeric_only=True)
                cmax = corr.abs().where(~np.eye(len(num), dtype=bool)).stack().sort_values(ascending=False)
                if len(cmax) > 0:
                    (c1, c2) = cmax.index[0]
                    figs.append(px.scatter(d, x=c1, y=c2, title=f"Top correlated pair: {c1} vs {c2}"))
            except Exception:
                pass

            try:
                corr = d[num].corr(numeric_only=True)
                top = corr.abs().mean().sort_values(ascending=False).head(10).index.tolist()
                heat = d[top].corr(numeric_only=True)
                figs.append(px.imshow(heat, text_auto=True, title="Correlation heatmap (top 10 numeric)"))
            except Exception:
                pass

        return figs[:max_plots]

    def build_report(self, df: pd.DataFrame, raw_df: Optional[pd.DataFrame], steps: List[str], target: Optional[str]) -> str:
        self._v(df)

        def s(df_):
            return {"rows": int(len(df_)), "cols": int(df_.shape[1]), "missing": int(df_.isnull().sum().sum()), "duplicates": int(df_.duplicated().sum())}

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        raw_block = ""
        if raw_df is not None and not raw_df.empty:
            rs = s(raw_df)
            raw_block = f"- Raw rows: {rs['rows']}\n- Raw cols: {rs['cols']}\n- Raw missing: {rs['missing']}\n- Raw duplicates: {rs['duplicates']}\n"

        cs = s(df)
        tgt_md = target if target else "(not selected)"

        missing_by_col = df.isnull().sum().sort_values(ascending=False).head(15).reset_index()
        missing_by_col.columns = ["column", "missing"]
        miss_md = missing_by_col.to_markdown(index=False)

        steps_md = "\n".join([f"- {x}" for x in steps]) if steps else "- (none)"
        head_md = df.head(8).to_markdown(index=False)

        metrics_block = ""
        if self.last_baseline is not None:
            metrics_block += f"\n## Baseline\n- {self.last_baseline}\n"
        if self.last_metrics is not None:
            metrics_block += f"\n## Model\n- {self.last_metrics}\n"

        rep = f"""# Data Science Project Report

Generated: {now}

## Dataset summary
Target: **{tgt_md}**

{raw_block}- Current rows: {cs['rows']}
- Current cols: {cs['cols']}
- Current missing: {cs['missing']}
- Current duplicates: {cs['duplicates']}

## Top missing columns (top 15)
{miss_md}

## Cleaning / engineering steps
{steps_md}

## Sample rows (first 8)
{head_md}
{metrics_block}
"""
        return rep.strip()

    def export_code(self, df: pd.DataFrame, target: str, algo: str, params: Dict[str, Any]) -> str:
        self._v(df)
        ps = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        cols = list(df.columns)

        if algo == "Random Forest":
            cls, reg = "RandomForestClassifier", "RandomForestRegressor"
        elif algo == "KNN":
            cls, reg = "KNeighborsClassifier", "KNeighborsRegressor"
        elif algo == "Decision Tree":
            cls, reg = "DecisionTreeClassifier", "DecisionTreeRegressor"
        else:
            cls, reg = "LogisticRegression", "LinearRegression"

        return f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# df = pd.read_csv("your_dataset.csv")
# Columns: {cols}

target = {repr(target)}
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = X.select_dtypes(include=["number"]).columns
cat = X.select_dtypes(include=["object", "category", "bool"]).columns

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), cat),
])

is_classification = (y.dtype == "object") or (y.nunique(dropna=True) < 10)

if is_classification:
    model = {cls}({ps})
else:
    model = {reg}({ps})

pipe = Pipeline([("preprocessor", pre), ("model", model)])
pipe.fit(X_train, y_train)
print("Score:", pipe.score(X_test, y_test))
""".strip()


def test_add_presence_flag():
    tk = DataToolkit()
    df = pd.DataFrame({"Cabin": ["C23", None, "", "  ", "E10"]})
    out = tk.add_presence_flag(df, "Cabin", "HasCabin")
    assert out["HasCabin"].tolist() == [1, 0, 0, 0, 1]


def test_clip_outliers_iqr():
    tk = DataToolkit()
    df = pd.DataFrame({"x": [1, 2, 3, 100]})
    out = tk.clip_outliers_iqr(df, "x", factor=1.5)
    assert float(out["x"].max()) < 100


def test_rename_drop():
    tk = DataToolkit()
    df = pd.DataFrame({"a": [1], "b": [2]})
    df2 = tk.rename_column(df, "a", "a2")
    assert "a2" in df2.columns and "a" not in df2.columns
    df3 = tk.drop_columns(df2, ["b"])
    assert "b" not in df3.columns
