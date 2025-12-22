import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from logic import AdvancedDataScientist

# --- Page Config ---
st.set_page_config(
    page_title="AI Data Scientist Pro",
    layout="wide",
    page_icon="🤖"
)

# --- Session State Initialization ---
if 'ads' not in st.session_state:
    st.session_state.ads = AdvancedDataScientist()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 AI Data Scientist Pro</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get one at aistudio.google.com")
    
    st.divider()
    
    uploaded_file = st.file_uploader("📁 Upload Dataset", type=['csv', 'xlsx', 'parquet'])
    
    if uploaded_file:
        if st.session_state.data is None:
            with st.spinner("Loading data..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                    
                    # Performance optimization for large datasets
                    if len(df) > 10000:
                        df = df.sample(10000, random_state=42)
                        st.info("⚠️ Sampled 10,000 rows for better performance")
                    
                    st.session_state.data = df
                    st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

# --- Main Logic ---
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Dataset Info Header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Rows", f"{len(df):,}")
    col2.metric("📈 Columns", len(df.columns))
    col3.metric("🔍 Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Overview", 
        "🔍 Smart Analysis", 
        "🤖 ML Training", 
        "📊 Visualization",
        "💬 AI Assistant"
    ])
    
    # 1. Data Overview
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    # 2. Smart Analysis
    with tab2:
        st.subheader("🔍 Smart Data Analysis")
        target_col = st.selectbox("Select Target Variable (What you want to predict)", options=df.columns)
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                analysis = st.session_state.ads.analyze_data_relationships(df)
                st.session_state.analysis_results = analysis
                task_type = st.session_state.ads.detect_task_type(df, target_col)
                suggestions = st.session_state.ads.suggest_algorithm(df, target_col, task_type)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"Detected Task: **{task_type.upper()}**")
                    if 'correlation_matrix' in analysis:
                        st.write("**Correlation Heatmap**")
                        fig = px.imshow(analysis['correlation_matrix'], color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.write("**Recommended Algorithms:**")
                    for algo in suggestions[task_type]:
                        st.markdown(f"""
                        <div class="model-card">
                            <strong>{algo['name']}</strong><br>{algo['reason']}
                        </div>
                        """, unsafe_allow_html=True)

    # 3. ML Training
    with tab3:
        st.subheader("🤖 Train a Model")
        
        if 'target_col' not in locals():
            target_col = st.selectbox("Select Target", options=df.columns, key='train_target')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            task_type = st.session_state.ads.detect_task_type(df, target_col)
            st.write(f"Task: **{task_type}**")
            
            # Dynamic Algorithm List
            if task_type == 'classification':
                algos = ["Random Forest", "Logistic Regression", "Gradient Boosting", "K-Nearest Neighbors", "Decision Tree"]
                if hasattr(st.session_state.ads, 'HAS_XGB') or True: # Check available via logic.py
                     algos.append("XGBoost")
            else:
                algos = ["Random Forest Regressor", "Linear Regression", "Gradient Boosting Regressor", "Ridge", "Lasso"]
                if hasattr(st.session_state.ads, 'HAS_XGB') or True:
                     algos.append("XGBoost Regressor")
            
            selected_algo = st.selectbox("Select Algorithm", algos)
            
            # Hyperparams
            suggestions = st.session_state.ads.get_hyperparameter_suggestions(selected_algo, df.shape)
            defaults = suggestions.get('default', {})
            
            hyperparams = {}
            if 'Random Forest' in selected_algo:
                hyperparams['n_estimators'] = st.slider("n_estimators", 10, 500, defaults.get('n_estimators', 100))
                hyperparams['max_depth'] = st.slider("max_depth", 1, 50, defaults.get('max_depth', 10))
            elif 'K-Nearest' in selected_algo:
                hyperparams['n_neighbors'] = st.slider("Neighbors", 1, 20, 5)

        with col2:
            if st.button("Start Training", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        model, metrics = st.session_state.ads.train_model(
                            df, target_col, selected_algo, hyperparams, task_type
                        )
                        st.success("Training Complete!")
                        st.write("### 📊 Metrics")
                        st.dataframe(pd.DataFrame([metrics]).T)
                        
                        if st.session_state.ads.feature_importance is not None:
                            st.write("### Feature Importance")
                            fig = px.bar(st.session_state.ads.feature_importance.head(10), 
                                       x='importance', y='feature', orientation='h')
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Training failed: {e}")

    # 4. Visualization
    with tab4:
        st.subheader("📊 Visualization Studio")
        viz_type = st.selectbox("Chart Type", ["Scatter", "Histogram", "Box Plot", "Bar Chart"])
        x_col = st.selectbox("X Axis", df.columns)
        y_col = st.selectbox("Y Axis", df.columns)
        color_col = st.selectbox("Color By", [None] + list(df.columns))
        
        if viz_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        elif viz_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col)
        elif viz_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col)
        elif viz_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col)
            
        st.plotly_chart(fig, use_container_width=True)

    # 5. AI Assistant
    with tab5:
        st.subheader("💬 AI Data Assistant")
        if api_key:
            genai.configure(api_key=api_key)
            user_query = st.text_area("Ask a question about your data:")
            
            if st.button("Ask AI"):
                if user_query:
                    # Construct context
                    context = f"""
                    Dataset Info: {len(df)} rows, columns: {list(df.columns)}.
                    Target Variable: {target_col if 'target_col' in locals() else 'None'}.
                    User Question: {user_query}
                    """
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        response = model.generate_content(context)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")
        else:
            st.warning("Please enter your Gemini API Key in the sidebar.")

else:
    st.info("👈 Please upload a dataset (CSV/Excel) from the sidebar to begin.")