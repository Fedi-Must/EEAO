import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, 
    Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings

# Robust imports for optional libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. XGBoost models will be unavailable.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not installed. LightGBM models will be unavailable.")

warnings.filterwarnings('ignore')

class AdvancedDataScientist:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.analysis_results = {}
        
    def detect_task_type(self, df, target):
        """Intelligently detect task type"""
        nunique = df[target].nunique()
        dtype = df[target].dtype
        
        if nunique < 10 or dtype == 'object' or str(dtype) == 'category':
            if nunique == 2:
                return 'classification' # binary
            return 'classification' # multiclass
        elif (dtype == 'int64' or dtype == 'float64') and nunique > 20:
            return 'regression'
        else:
            return 'classification' # Default fallback
    
    def analyze_data_relationships(self, df):
        """Analyze correlations and relationships between features"""
        analysis = {}
        
        # Numeric correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            analysis['correlation_matrix'] = correlation_matrix
            
            # Find strong correlations (absolute > 0.7)
            strong_corrs = []
            cols = correlation_matrix.columns
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    val = correlation_matrix.iloc[i, j]
                    if abs(val) > 0.7:
                        strong_corrs.append({
                            'feature1': cols[i],
                            'feature2': cols[j],
                            'correlation': val
                        })
            analysis['strong_correlations'] = strong_corrs
        
        return analysis
    
    def suggest_algorithm(self, df, target, task_type):
        """Suggest best algorithms based on data characteristics"""
        suggestions = {
            'classification': [],
            'regression': [],
            'feature_selection': []
        }
        
        n_samples = len(df)
        
        # Classification suggestions
        if task_type == 'classification':
            if n_samples < 1000:
                suggestions['classification'].extend([
                    {'name': 'Logistic Regression', 'reason': 'Good baseline for small datasets'},
                    {'name': 'Random Forest', 'reason': 'Robust to overfitting on small data'}
                ])
            else:
                suggestions['classification'].extend([
                    {'name': 'XGBoost', 'reason': 'High performance on larger tabular data'},
                    {'name': 'Gradient Boosting', 'reason': 'High accuracy through boosting'}
                ])
        
        # Regression suggestions
        elif task_type == 'regression':
            if n_samples < 1000:
                suggestions['regression'].extend([
                    {'name': 'Linear Regression', 'reason': 'Simple and interpretable'},
                    {'name': 'Ridge', 'reason': 'Handles multicollinearity well'}
                ])
            else:
                suggestions['regression'].extend([
                    {'name': 'Random Forest Regressor', 'reason': 'Captures non-linear patterns'},
                    {'name': 'XGBoost Regressor', 'reason': 'State-of-the-art accuracy'}
                ])
                
        return suggestions
    
    def get_hyperparameter_suggestions(self, algorithm_name, X_shape):
        """Suggest hyperparameters"""
        suggestions = {}
        n_samples = X_shape[0]
        
        if 'Random Forest' in algorithm_name:
            suggestions['default'] = {
                'n_estimators': 100,
                'max_depth': 10 if n_samples > 1000 else 5,
                'min_samples_split': 2
            }
        elif 'K-Nearest' in algorithm_name:
            suggestions['default'] = {
                'n_neighbors': 5,
                'weights': 'uniform'
            }
        else:
            suggestions['default'] = {}
            
        return suggestions
    
    def train_model(self, df, target, algorithm, hyperparams=None, task_type=None):
        """Train a specific model with given hyperparameters"""
        if task_type is None:
            task_type = self.detect_task_type(df, target)
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Preprocessing Pipelines
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get Model
        model_instance = self._get_model_instance(algorithm, task_type, hyperparams)
        
        # Create Pipeline
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_instance)
        ])
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict & Evaluate
        preds = clf.predict(X_test)
        metrics = self._calculate_metrics(y_test, preds, task_type)
        
        # Extract Feature Importance (if applicable)
        try:
            # Access the model step inside the pipeline
            final_model = clf.named_steps['model']
            if hasattr(final_model, 'feature_importances_'):
                # Get feature names after one-hot encoding
                ohe_cols = clf.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
                all_cols = list(numeric_features) + list(ohe_cols)
                
                if len(all_cols) == len(final_model.feature_importances_):
                    self.feature_importance = pd.DataFrame({
                        'feature': all_cols,
                        'importance': final_model.feature_importances_
                    }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            self.feature_importance = None
            
        self.model = clf
        return clf, metrics
    
    def _calculate_metrics(self, y_true, y_pred, task_type):
        """Calculate performance metrics"""
        metrics = {}
        if task_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
            metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
        else:
            metrics['R2 Score'] = r2_score(y_true, y_pred)
            metrics['MSE'] = mean_squared_error(y_true, y_pred)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        return metrics

    def _get_model_instance(self, algorithm, task_type, hyperparams=None):
        """Get model instance based on algorithm name and task"""
        if hyperparams is None: hyperparams = {}
        
        # CLASSIFICATION MODELS
        if task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(**hyperparams),
                'Random Forest': RandomForestClassifier(**hyperparams),
                'Gradient Boosting': GradientBoostingClassifier(**hyperparams),
                'K-Nearest Neighbors': KNeighborsClassifier(**hyperparams),
                'Decision Tree': DecisionTreeClassifier(**hyperparams),
                'Support Vector Machine': SVC(**hyperparams),
                'AdaBoost': AdaBoostClassifier(**hyperparams),
                'Gaussian Naive Bayes': GaussianNB(**hyperparams)
            }
            if HAS_XGB:
                models['XGBoost'] = xgb.XGBClassifier(**hyperparams, use_label_encoder=False, eval_metric='logloss')
            if HAS_LGB:
                models['LightGBM'] = lgb.LGBMClassifier(**hyperparams)
                
            return models.get(algorithm, RandomForestClassifier(**hyperparams))

        # REGRESSION MODELS
        elif task_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(**hyperparams),
                'Random Forest Regressor': RandomForestRegressor(**hyperparams),
                'Gradient Boosting Regressor': GradientBoostingRegressor(**hyperparams),
                'KNN Regression': KNeighborsRegressor(**hyperparams),
                'Decision Tree Regressor': DecisionTreeRegressor(**hyperparams),
                'Support Vector Regression': SVR(**hyperparams),
                'Ridge': Ridge(**hyperparams),
                'Lasso': Lasso(**hyperparams)
            }
            if HAS_XGB:
                models['XGBoost Regressor'] = xgb.XGBRegressor(**hyperparams)
            if HAS_LGB:
                models['LightGBM Regressor'] = lgb.LGBMRegressor(**hyperparams)
                
            return models.get(algorithm, RandomForestRegressor(**hyperparams))
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")