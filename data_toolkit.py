from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from toolkit_cleaning import CleaningMixin
from toolkit_core import ToolkitCoreMixin
from toolkit_modeling import ModelingMixin
from toolkit_reporting import ReportingMixin
from toolkit_visualization import VisualizationMixin


class DataToolkit(
    ToolkitCoreMixin,
    ModelingMixin,
    CleaningMixin,
    VisualizationMixin,
    ReportingMixin,
):
    """Utility class for cleaning, modeling, reporting, and plotting dataset workflows."""

    def __init__(self) -> None:
        """Initialize runtime model artifacts and metadata placeholders."""
        self.model_pipeline: Optional[Pipeline] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.task_type: Optional[str] = None
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.last_baseline: Optional[Dict[str, Any]] = None
        self.last_loaded_meta: Optional[Dict[str, Any]] = None
