from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from toolkit_constants import CLASSIFICATION_CARDINALITY_THRESHOLD


class ToolkitCoreMixin:
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Ensure an input dataframe exists and contains at least one row."""
        if df is None or df.empty:
            raise ValueError("Dataset is empty.")

    @staticmethod
    def _validate_target_column(df: pd.DataFrame, target: str) -> None:
        """Ensure the requested target column is present in the dataframe."""
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")

    @staticmethod
    def _is_classification_target(target_series: pd.Series) -> bool:
        """Heuristically classify target type from dtype and cardinality."""
        if target_series.dtype == "object":
            return True
        return target_series.nunique(dropna=True) < CLASSIFICATION_CARDINALITY_THRESHOLD

    @staticmethod
    def _make_one_hot_encoder() -> OneHotEncoder:
        """Create a OneHotEncoder compatible with older/newer sklearn versions."""
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    @staticmethod
    def _dataset_stats(df: pd.DataFrame) -> Dict[str, int]:
        """Return standard dataset statistics used across the app."""
        return {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        }
