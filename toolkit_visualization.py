from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px


class VisualizationMixin:
    def suggest_plots(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        max_plots: int = 3,
    ) -> List[Any]:
        """Generate a small set of automatic exploratory Plotly figures."""
        self._validate_dataframe(df)
        figures: List[Any] = []
        working_df = df.copy()

        if target and target in working_df.columns:
            try:
                figures.append(px.histogram(working_df, x=target, title="Target distribution"))
            except Exception:
                pass

        numeric_columns = working_df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) >= 2:
            try:
                correlation = working_df[numeric_columns].corr(numeric_only=True)
                correlation_pairs = (
                    correlation.abs()
                    .where(~np.eye(len(numeric_columns), dtype=bool))
                    .stack()
                    .sort_values(ascending=False)
                )
                if len(correlation_pairs) > 0:
                    col_a, col_b = correlation_pairs.index[0]
                    figures.append(
                        px.scatter(
                            working_df,
                            x=col_a,
                            y=col_b,
                            title=f"Top correlated pair: {col_a} vs {col_b}",
                        )
                    )
            except Exception:
                pass

            try:
                correlation = working_df[numeric_columns].corr(numeric_only=True)
                top_columns = (
                    correlation.abs().mean().sort_values(ascending=False).head(10).index.tolist()
                )
                heatmap_source = working_df[top_columns].corr(numeric_only=True)
                figures.append(
                    px.imshow(
                        heatmap_source,
                        text_auto=True,
                        title="Correlation heatmap (top 10 numeric)",
                    )
                )
            except Exception:
                pass

        return figures[:max_plots]
