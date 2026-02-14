from __future__ import annotations

from typing import List

import pandas as pd


class CleaningMixin:
    def fill_missing(self, df: pd.DataFrame, col: str, strat: str) -> pd.DataFrame:
        """Fill missing values in a column using the selected strategy."""
        self._validate_dataframe(df)
        if col not in df.columns:
            raise ValueError("Column not found.")

        out = df.copy()
        series = out[col]
        strategy = strat

        if strategy == "Auto (Median/Mode)":
            strategy = "Median" if pd.api.types.is_numeric_dtype(series) else "Mode"

        if strategy == "Mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Mean requires numeric column.")
            out[col] = series.fillna(float(series.mean()))
        elif strategy == "Median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Median requires numeric column.")
            out[col] = series.fillna(float(series.median()))
        elif strategy == "Mode":
            mode_values = series.mode(dropna=True)
            out[col] = series.fillna(mode_values.iloc[0] if not mode_values.empty else None)
        elif strategy == "Zero":
            out[col] = series.fillna(0)
        elif strategy == "Unknown":
            if pd.api.types.is_numeric_dtype(series):
                raise ValueError("Unknown is for categorical columns.")
            out[col] = series.fillna("Unknown")
        else:
            raise ValueError("Unknown strategy.")

        return out

    def drop_missing_rows(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Drop rows where the selected column is missing."""
        self._validate_dataframe(df)
        if col not in df.columns:
            raise ValueError("Column not found.")
        return df.dropna(subset=[col]).copy()

    def drop_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Drop one or more columns after validating they all exist."""
        self._validate_dataframe(df)
        missing_columns = [column for column in cols if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found: {missing_columns}")
        return df.drop(columns=cols).copy()

    def rename_column(self, df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
        """Rename a column with collision/validity checks."""
        self._validate_dataframe(df)
        if old not in df.columns:
            raise ValueError("Old column not found.")
        if new in df.columns and new != old:
            raise ValueError("New name already exists.")
        return df.rename(columns={old: new}).copy()

    def clip_outliers_iqr(self, df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
        """Clip numeric outliers in a column using IQR bounds."""
        self._validate_dataframe(df)
        if col not in df.columns:
            raise ValueError("Column not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError("Requires numeric column.")

        series = df[col].astype(float)
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        out = df.copy()
        out[col] = series.clip(lower_bound, upper_bound)
        return out

    def add_presence_flag(self, df: pd.DataFrame, source_col: str, new_col: str) -> pd.DataFrame:
        """Create a 0/1 feature indicating whether source values are present."""
        self._validate_dataframe(df)
        if source_col not in df.columns:
            raise ValueError("Source column not found.")
        if not new_col or not str(new_col).strip():
            raise ValueError("New column name cannot be empty.")
        if new_col in df.columns:
            raise ValueError("New column already exists.")

        out = df.copy()
        series = out[source_col]
        is_present = series.notna()
        if series.dtype == "object" or str(series.dtype).startswith("string"):
            is_present = is_present & series.astype(str).str.strip().ne("")
        out[new_col] = is_present.astype(int)
        return out
