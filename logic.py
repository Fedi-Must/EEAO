import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, r2_score, 
                             classification_report, confusion_matrix)

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

# Try importing optional libraries without crashing
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class DataToolkit:
    def __init__(self):
        self.model_pipeline = None
        self.feature_importance = None
        self.task_type = None
        
    def clean_missing_values(self, df, column, strategy):
        """Simple imputation wrapper"""
        df_clean = df.copy()
        if strategy == "Mean":
            val = df_clean[column].mean()
            df_clean[column] = df_clean[column].fillna(val)
        elif strategy == "Median":
            val = df_clean[column].median()
            df_clean[column] = df_clean[column].fillna(val)
        elif strategy == "Mode":
            val = df_clean[column].mode()[0]
            df_clean[column] = df_clean[column].fillna(val)
        elif strategy == "Drop rows":
            df_clean = df_clean.dropna(subset=[column])
        elif strategy == "Fill with zero":
            df_clean[column] = df_clean[column].fillna(0)
        return df_clean

    def detect_task_type(self, df, target):
        """Heuristic to decide Classification vs Regression"""
        if df[target].nunique() < 10 or df[target].dtype == 'object':
            return 'classification'
        return 'regression'

    def get_preprocessor(self, X):
        """Builds the column transformer for the pipeline"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, numeric_features),
                ('cat', cat_transformer, categorical_features)
            ])

    def compare_models(self, df, target):
        """AutoML Loop - Uses Cross Validation to prevent overfitting"""
        task = self.detect_task_type(df, target)
        X = df.drop(columns=[target])
        y = df[target]

        # Define candidate models
        if task == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=50),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier()
            }
            metric = 'accuracy'
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=50),
                'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
                'Ridge': Ridge()
            }
            metric = 'r2'

        results = []
        
        # Preprocessing setup
        preprocessor = self.get_preprocessor(X)

        for name, model in models.items():
            # Create a full pipeline for every model to prevent data leakage
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Cross Validation (Standard Master's technique)
            cv_scores = cross_val_score(clf, X, y, cv=5, scoring=metric)
            results.append({
                'Model': name,
                'Score': cv_scores.mean(),
                'Std': cv_scores.std()
            })

        return pd.DataFrame(results).sort_values(by='Score', ascending=False)

    def train_model(self, df, target, algo_name, params, task_type, test_size):
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        preprocessor = self.get_preprocessor(X_train)
        
        # Select Model Class
        if task_type == 'classification':
            if 'Random Forest' in algo_name: model = RandomForestClassifier(**params)
            elif 'KNN' in algo_name: model = KNeighborsClassifier(**params)
            elif 'Logistic' in algo_name: model = LogisticRegression(**params)
            else: model = DecisionTreeClassifier(**params)
        else:
            if 'Random Forest' in algo_name: model = RandomForestRegressor(**params)
            elif 'KNN' in algo_name: model = KNeighborsRegressor(**params)
            elif 'Linear' in algo_name: model = LinearRegression(**params)
            else: model = Ridge(**params)

        # Build and Fit Pipeline
        self.model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        self.model_pipeline.fit(X_train, y_train)
        
        # Predictions
        preds = self.model_pipeline.predict(X_test)
        
        # Calculate Metrics
        metrics = {}
        if task_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y_test, preds)
            metrics['F1 Score'] = f1_score(y_test, preds, average='weighted')
        else:
            metrics['R2 Score'] = r2_score(y_test, preds)
            metrics['MSE'] = mean_squared_error(y_test, preds)
            
        # Try to get feature importance (Robust method)
        self.feature_importance = None
        if hasattr(model, 'feature_importances_'):
            try:
                # Extract feature names from the transformer
                num_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()
                all_cols = np.concatenate([num_cols, cat_cols])
                
                if len(all_cols) == len(model.feature_importances_):
                    self.feature_importance = pd.DataFrame({
                        'feature': all_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
            except:
                pass # Silently fail if feature names don't match (common sklearn issue)

        return self.model_pipeline, metrics, X_test, y_test, preds

    def predict(self, input_data_dict):
        """Predicts on single row input"""
        if not self.model_pipeline:
            raise ValueError("Model not trained yet")
        input_df = pd.DataFrame([input_data_dict])
        return self.model_pipeline.predict(input_df)[0]

    def export_code(self, df, target, algo, params):
        """Generates reproducible python code"""
        param_str = ", ".join([f"{k}={v}" for k,v in params.items()])
        cols = list(df.columns)
        
        code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. Load Data
# df = pd.read_csv('your_dataset.csv')
# Columns expected: {cols}

target = '{target}'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocessing
num_features = X.select_dtypes(include=['number']).columns
cat_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
])

# 3. Model
# Algorithm: {algo}
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', {algo.replace(' ', '')}({param_str})) 
])

model.fit(X_train, y_train)
print("Score:", model.score(X_test, y_test))
"""
        return code