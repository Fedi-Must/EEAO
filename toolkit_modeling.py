from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from toolkit_constants import CLASSIFICATION, DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE, REGRESSION


class ModelingMixin:
    def detect_task_type(self, df: pd.DataFrame, target: str) -> str:
        """Infer whether the target should be treated as classification or regression."""
        self._validate_dataframe(df)
        self._validate_target_column(df, target)
        return CLASSIFICATION if self._is_classification_target(df[target]) else REGRESSION

    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipelines for numeric and categorical features."""
        numeric_columns = X.select_dtypes(include=["number"]).columns
        categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns

        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", self._make_one_hot_encoder()),
            ]
        )

        return ColumnTransformer(
            [("num", numeric_pipeline, numeric_columns), ("cat", categorical_pipeline, categorical_columns)],
            remainder="drop",
        )

    def _candidate_models(self, task: str) -> Tuple[Dict[str, Any], str]:
        """Return candidate estimators and the scoring metric for model comparison."""
        if task == CLASSIFICATION:
            return (
                {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=50, random_state=DEFAULT_RANDOM_STATE
                    ),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "Decision Tree": DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
                },
                "accuracy",
            )

        return (
            {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=50, random_state=DEFAULT_RANDOM_STATE
                ),
                "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
                "Ridge": Ridge(),
            },
            "r2",
        )

    def compare_models(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Cross-validate candidate models and return ranked performance summary."""
        self._validate_dataframe(df)
        task = self.detect_task_type(df, target)

        features = df.drop(columns=[target])
        labels = df[target]
        if features.shape[1] == 0:
            raise ValueError("No features found.")
        if labels.isna().any():
            raise ValueError("Target contains missing values. Fill or drop missing target rows first.")

        if task == CLASSIFICATION:
            class_counts = labels.value_counts(dropna=False)
            max_cv_folds = int(class_counts.min()) if not class_counts.empty else 0
            if max_cv_folds < 2:
                raise ValueError("Need at least 2 rows in each target class to compare models.")
        else:
            max_cv_folds = int(len(features))
            if max_cv_folds < 2:
                raise ValueError("Need at least 2 rows to compare models.")

        cv_folds = min(DEFAULT_CV_FOLDS, max_cv_folds, int(len(features)))

        models, scoring = self._candidate_models(task)
        preprocessor = self.get_preprocessor(features)

        rows: List[Dict[str, float]] = []
        failures: List[str] = []
        for model_name, estimator in models.items():
            candidate = clone(estimator)
            if hasattr(candidate, "n_neighbors"):
                min_train_size = int(len(features)) - ((len(features) + cv_folds - 1) // cv_folds)
                safe_neighbors = max(1, min(int(getattr(candidate, "n_neighbors")), min_train_size))
                candidate.set_params(n_neighbors=safe_neighbors)

            pipeline = Pipeline([("preprocessor", preprocessor), ("model", candidate)])
            try:
                scores = cross_val_score(
                    pipeline,
                    features,
                    labels,
                    cv=cv_folds,
                    scoring=scoring,
                )
                rows.append(
                    {
                        "Model": model_name,
                        "Score": float(scores.mean()),
                        "Std": float(scores.std()),
                    }
                )
            except Exception as exc:
                failures.append(f"{model_name}: {exc}")

        if not rows:
            if failures:
                raise ValueError(f"Model comparison failed. {failures[0]}")
            raise ValueError("Model comparison failed for this dataset.")

        return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

    def _make_model(self, task: str, algo: str, params: Dict[str, Any]) -> Any:
        """Instantiate a concrete estimator from UI selections."""
        if task == CLASSIFICATION:
            if "Random Forest" in algo:
                return RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE, **params)
            if "KNN" in algo:
                return KNeighborsClassifier(**params)
            if "Linear/Logistic" in algo or "Logistic" in algo:
                return LogisticRegression(max_iter=1000, **params)
            return DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE, **params)

        if "Random Forest" in algo:
            return RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **params)
        if "KNN" in algo:
            return KNeighborsRegressor(**params)
        if "Linear/Logistic" in algo or "Linear" in algo:
            return LinearRegression(**params)
        if "Decision Tree" in algo:
            return DecisionTreeRegressor(random_state=DEFAULT_RANDOM_STATE, **params)
        return Ridge(**params)

    @staticmethod
    def _build_baseline_model(task: str) -> Any:
        """Create a dummy baseline model for fair metric comparison."""
        if task == CLASSIFICATION:
            return DummyClassifier(strategy="most_frequent")
        return DummyRegressor(strategy="mean")

    @staticmethod
    def _compute_metrics(task: str, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute task-appropriate evaluation metrics."""
        if task == CLASSIFICATION:
            return {
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "F1": float(f1_score(y_true, y_pred, average="weighted")),
            }
        return {
            "R2": float(r2_score(y_true, y_pred)),
            "MSE": float(mean_squared_error(y_true, y_pred)),
        }

    def train_with_baseline(
        self,
        df: pd.DataFrame,
        target: str,
        algo: str,
        params: Dict[str, Any],
        task_type: str,
        test_size: float,
    ) -> Dict[str, Any]:
        """Train selected model and baseline, then store metrics and artifacts."""
        self._validate_dataframe(df)
        self._validate_target_column(df, target)

        features = df.drop(columns=[target])
        labels = df[target]
        if features.shape[1] == 0:
            raise ValueError("No features found.")
        if labels.isna().any():
            raise ValueError("Target contains missing values. Fill or drop missing target rows first.")

        row_count = int(len(features))
        if row_count < 4:
            raise ValueError("Need at least 4 rows to train and evaluate a model.")

        if task_type not in {CLASSIFICATION, REGRESSION}:
            raise ValueError(f"Unknown task type: {task_type}")

        test_fraction = float(test_size)
        if test_fraction <= 0.0 or test_fraction >= 1.0:
            raise ValueError("test_size must be between 0 and 1.")

        if task_type == CLASSIFICATION:
            class_counts = labels.value_counts(dropna=False)
            if class_counts.min() < 2:
                raise ValueError("Need at least 2 rows in each target class to train/test split.")
            stratify_labels: Optional[pd.Series] = labels
        else:
            stratify_labels = None

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_fraction,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=stratify_labels,
        )
        preprocessor = self.get_preprocessor(X_train)

        baseline_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("model", self._build_baseline_model(task_type))]
        )
        baseline_pipeline.fit(X_train, y_train)
        baseline_metrics = self._compute_metrics(task_type, y_test, baseline_pipeline.predict(X_test))

        model = self._make_model(task_type, algo, params)
        model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        model_pipeline.fit(X_train, y_train)
        model_metrics = self._compute_metrics(task_type, y_test, model_pipeline.predict(X_test))

        self.model_pipeline = model_pipeline
        self.task_type = task_type
        self.last_metrics = {
            "model": model_metrics,
            "test_size": float(test_size),
            "algo": algo,
            "params": params,
        }
        self.last_baseline = {
            "baseline": baseline_metrics,
            "strategy": "most_frequent" if task_type == CLASSIFICATION else "mean",
        }
        self.feature_importance = self._feature_importance()

        return {"baseline_metrics": baseline_metrics, "model_metrics": model_metrics}

    def _feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importance or coefficient magnitude when supported."""
        if self.model_pipeline is None:
            return None

        preprocessor = self.model_pipeline.named_steps.get("preprocessor")
        model = self.model_pipeline.named_steps.get("model")
        if preprocessor is None or model is None:
            return None

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            return None

        importance_values: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_"):
            importance_values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importance_values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()

        if importance_values is None:
            return None

        row_count = min(len(feature_names), len(importance_values))
        feature_table = pd.DataFrame(
            {
                "feature": feature_names[:row_count],
                "importance": importance_values[:row_count],
            }
        )
        return feature_table.sort_values("importance", ascending=False).reset_index(drop=True)

    def predict(self, row: Dict[str, Any]) -> Any:
        """Run a single-row prediction using the trained or loaded pipeline."""
        if self.model_pipeline is None:
            raise ValueError("Model not trained/loaded.")
        return self.model_pipeline.predict(pd.DataFrame([row]))[0]

    def save_model(self, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
        """Serialize model pipeline and metadata to disk."""
        if self.model_pipeline is None:
            raise ValueError("No model to save.")
        payload = {
            "pipeline": self.model_pipeline,
            "task_type": self.task_type,
            "meta": meta or {},
            "metrics": self.last_metrics,
            "baseline": self.last_baseline,
        }
        joblib.dump(payload, path)

    def load_model(self, file_obj: Any) -> Dict[str, Any]:
        """Load a previously saved model payload or raw pipeline and return metadata."""
        payload = joblib.load(file_obj)
        if isinstance(payload, dict) and "pipeline" in payload:
            self.model_pipeline = payload["pipeline"]
            self.task_type = payload.get("task_type")
            self.last_metrics = payload.get("metrics")
            self.last_baseline = payload.get("baseline")
            loaded_meta = payload.get("meta")
            if not isinstance(loaded_meta, dict):
                loaded_meta = {}
        else:
            self.model_pipeline = payload
            self.task_type = None
            self.last_metrics = None
            self.last_baseline = None
            loaded_meta = {}

        self.last_loaded_meta = loaded_meta
        self.feature_importance = self._feature_importance()
        return loaded_meta
