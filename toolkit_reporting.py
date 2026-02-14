from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd


class ReportingMixin:
    def build_report(
        self,
        df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame],
        steps: List[str],
        target: Optional[str],
    ) -> str:
        """Build a markdown project report from current data, steps, and model stats."""
        self._validate_dataframe(df)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        target_markdown = target if target else "(not selected)"

        raw_block = ""
        if raw_df is not None and not raw_df.empty:
            raw_stats = self._dataset_stats(raw_df)
            raw_block = (
                f"- Raw rows: {raw_stats['rows']}\n"
                f"- Raw cols: {raw_stats['cols']}\n"
                f"- Raw missing: {raw_stats['missing']}\n"
                f"- Raw duplicates: {raw_stats['duplicates']}\n"
            )

        current_stats = self._dataset_stats(df)
        missing_by_col = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
            .rename(columns={"index": "column", 0: "missing"})
        )
        missing_table = missing_by_col.to_markdown(index=False)

        steps_markdown = "\n".join(f"- {step}" for step in steps) if steps else "- (none)"
        sample_rows = df.head(8).to_markdown(index=False)

        metrics_block = ""
        if self.last_baseline is not None:
            metrics_block += f"\n## Baseline\n- {self.last_baseline}\n"
        if self.last_metrics is not None:
            metrics_block += f"\n## Model\n- {self.last_metrics}\n"

        report = f"""# Data Science Project Report

Generated: {now}

## Dataset summary
Target: **{target_markdown}**

{raw_block}- Current rows: {current_stats['rows']}
- Current cols: {current_stats['cols']}
- Current missing: {current_stats['missing']}
- Current duplicates: {current_stats['duplicates']}

## Top missing columns (top 15)
{missing_table}

## Cleaning / engineering steps
{steps_markdown}

## Sample rows (first 8)
{sample_rows}
{metrics_block}
"""
        return report.strip()

    def export_code(self, df: pd.DataFrame, target: str, algo: str, params: Dict[str, object]) -> str:
        """Export a runnable training script that mirrors the current modeling choices."""
        self._validate_dataframe(df)
        param_string = ", ".join([f"{key}={repr(value)}" for key, value in params.items()])
        columns = list(df.columns)

        if algo == "Random Forest":
            classifier_name, regressor_name = "RandomForestClassifier", "RandomForestRegressor"
        elif algo == "KNN":
            classifier_name, regressor_name = "KNeighborsClassifier", "KNeighborsRegressor"
        elif algo == "Decision Tree":
            classifier_name, regressor_name = "DecisionTreeClassifier", "DecisionTreeRegressor"
        else:
            classifier_name, regressor_name = "LogisticRegression", "LinearRegression"

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
# Columns: {columns}

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
    model = {classifier_name}({param_string})
else:
    model = {regressor_name}({param_string})

pipe = Pipeline([("preprocessor", pre), ("model", model)])
pipe.fit(X_train, y_train)
print("Score:", pipe.score(X_test, y_test))
""".strip()
