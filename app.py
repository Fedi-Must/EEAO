import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from logic import DataToolkit

# --- Page Config ---
st.set_page_config(page_title="Data Science Project Toolkit", layout="wide", page_icon="🧪")

# --- Session State ---
if 'toolkit' not in st.session_state:
    st.session_state.toolkit = DataToolkit()
if 'data' not in st.session_state:
    st.session_state.data = None

# --- Custom Styling (Minimalist) ---
st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #333; text-align: center; margin-bottom: 1rem; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧪 Data Science Project Toolkit</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Project Setup")
    api_key = st.text_input("Gemini API Key (Optional)", type="password")
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file and st.session_state.data is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success("Data Loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.info("Features:\n- Auto-Cleaning\n- KNN/RF/Regression\n- Model Comparison\n- Code Export")

# --- Main App ---
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Simple Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Cleaning", "AutoML & Training", "Prediction", "AI Help"])

    # 1. Overview
    with tab1:
        st.dataframe(df.head())
        st.write("### Column Stats")
        st.write(df.describe())
        
        st.write("### Visualization")
        col_x = st.selectbox("X Axis", df.columns)
        col_y = st.selectbox("Y Axis", df.columns)
        kind = st.selectbox("Type", ["Scatter", "Box", "Bar"])
        
        if kind == "Scatter": fig = px.scatter(df, x=col_x, y=col_y)
        elif kind == "Box": fig = px.box(df, x=col_x, y=col_y)
        else: fig = px.bar(df, x=col_x, y=col_y)
        st.plotly_chart(fig, use_container_width=True)

    # 2. Cleaning
    with tab2:
        st.subheader("Data Cleaning")
        col_to_clean = st.selectbox("Select Column to Clean", df.columns)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Fill Missing (Median/Mode)"):
                strategy = "Median" if pd.api.types.is_numeric_dtype(df[col_to_clean]) else "Mode"
                st.session_state.data = st.session_state.toolkit.clean_missing_values(df, col_to_clean, strategy)
                st.success(f"Filled {col_to_clean} with {strategy}")
                st.rerun()
        with c2:
            if st.button("Drop Rows with Missing"):
                st.session_state.data = st.session_state.toolkit.clean_missing_values(df, col_to_clean, "Drop rows")
                st.success(f"Dropped missing in {col_to_clean}")
                st.rerun()

    # 3. Training
    with tab3:
        target = st.selectbox("Target Variable", df.columns)
        task_type = st.session_state.toolkit.detect_task_type(df, target)
        st.info(f"Detected Task: {task_type.upper()}")
        
        mode = st.radio("Mode", ["Compare Models (AutoML)", "Train Specific Model"])
        
        if mode == "Compare Models (AutoML)":
            if st.button("Run Tournament"):
                with st.spinner("Running Cross-Validation..."):
                    leaderboard = st.session_state.toolkit.compare_models(df, target)
                    st.dataframe(leaderboard.style.highlight_max(axis=0, color='lightgreen'))
                    
        else:
            algo = st.selectbox("Algorithm", ["Random Forest", "KNN", "Linear/Logistic Regression", "Decision Tree"])
            
            # Simple Params
            params = {}
            if "Random Forest" in algo:
                params['n_estimators'] = st.slider("Trees", 10, 200, 100)
            if "KNN" in algo:
                params['n_neighbors'] = st.slider("Neighbors (K)", 1, 20, 5)
            
            if st.button("Train Model"):
                model, metrics, _, _, preds = st.session_state.toolkit.train_model(df, target, algo, params, task_type, 0.2)
                st.write("### Metrics")
                st.json(metrics)
                
                if st.session_state.toolkit.feature_importance is not None:
                    st.write("### Feature Importance")
                    st.bar_chart(st.session_state.toolkit.feature_importance.set_index('feature'))
                
                # Export Code
                st.subheader("Get Python Code")
                code = st.session_state.toolkit.export_code(df, target, algo, params)
                st.code(code, language='python')

    # 4. Prediction
    with tab4:
        st.subheader("Predict New Data")
        if st.session_state.toolkit.model_pipeline is None:
            st.warning("Train a model in Tab 3 first.")
        else:
            input_data = {}
            # Dynamically generate form based on feature columns
            features = [c for c in df.columns if c != target]
            
            cols = st.columns(3)
            for i, col in enumerate(features):
                with cols[i % 3]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_data[col] = st.number_input(col, value=float(df[col].mean()))
                    else:
                        input_data[col] = st.selectbox(col, df[col].unique())
            
            if st.button("Predict"):
                result = st.session_state.toolkit.predict(input_data)
                st.success(f"Prediction: {result}")

    # 5. AI Assistant
    with tab5:
        if api_key:
            genai.configure(api_key=api_key)
            q = st.text_input("Ask about your data:")
            if q:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(f"Data context: {df.head().to_string()}. User question: {q}")
                st.write(response.text)
        else:
            st.info("Add Gemini API Key in sidebar")