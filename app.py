"""
StreamlitAutoML Web Application

Interactive web interface for the AutoML system using Streamlit.
Provides data upload, EDA visualization, model training, and results analysis.
"""

import io
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configure Streamlit page
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional UI Design System
st.markdown(
    """
<style>
/* Design System Variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    --card-shadow-hover: 0 12px 48px rgba(0, 0, 0, 0.15);
    --spacing-xs: 0.5rem;
    --spacing-sm: 1rem;
    --spacing-md: 1.5rem;
    --spacing-lg: 2rem;
    --spacing-xl: 3rem;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
}

/* Global Enhancements */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Animated Header */
.main-header {
    background: var(--primary-gradient);
    padding: var(--spacing-lg);
    border-radius: var(--radius-xl);
    color: white;
    text-align: center;
    box-shadow: var(--card-shadow);
    margin-bottom: var(--spacing-lg);
    animation: slideDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

@keyframes slideDown {
    from { 
        transform: translateY(-30px);
        opacity: 0;
    }
    to { 
        transform: translateY(0);
        opacity: 1;
    }
}

/* Premium Card Styles */
.metric-card {
    background: white;
    color: #333333;
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    box-shadow: var(--card-shadow);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    margin: var(--spacing-sm) 0;
    border: 1px solid rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--primary-gradient);
    transition: width 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--card-shadow-hover);
}

.metric-card:hover::before {
    width: 100%;
    opacity: 0.05;
}

/* Info & Alert Boxes */
.info-box {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    border-left: 4px solid #4facfe;
    margin: var(--spacing-sm) 0;
    box-shadow: 0 2px 8px rgba(79, 172, 254, 0.15);
    backdrop-filter: blur(10px);
}

.success-box {
    background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    border-left: 4px solid #11998e;
    margin: var(--spacing-sm) 0;
    box-shadow: 0 2px 8px rgba(17, 153, 142, 0.15);
}

.warning-box {
    background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
    padding: var(--spacing-md);
    border-radius: var(--radius-md);
    border-left: 4px solid #f5576c;
    margin: var(--spacing-sm) 0;
    box-shadow: 0 2px 8px rgba(245, 87, 108, 0.15);
}

/* Progress Animation */
.stProgress > div > div > div > div {
    background: var(--primary-gradient) !important;
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Professional Sidebar */
[data-testid="stSidebar"] {
    background: var(--dark-gradient);
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.1);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #ECF0F1;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: white;
}

[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
    color: white;
}

[data-testid="stSidebar"] [data-baseweb="radio"] {
    background: rgba(255,255,255,0.05);
    border-radius: var(--radius-md);
    padding: var(--spacing-xs);
    margin: var(--spacing-xs) 0;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.1);
    transform: translateX(4px);
}

/* Premium Buttons */
.stButton > button {
    background: var(--primary-gradient);
    color: white;
    border: none;
    border-radius: var(--radius-lg);
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover::before {
    width: 300px;
    height: 300px;
}

.stButton > button:active {
    transform: translateY(0);
}

/* Data Table Enhancement */
.dataframe {
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--card-shadow);
    border: 1px solid rgba(0,0,0,0.05);
}

.dataframe thead tr {
    background: var(--primary-gradient);
    color: white;
}

.dataframe tbody tr:hover {
    background: rgba(102, 126, 234, 0.05);
}

/* Chart Containers */
.chart-container {
    background: white;
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    box-shadow: var(--card-shadow);
    margin: var(--spacing-sm) 0;
    border: 1px solid rgba(0,0,0,0.05);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.chart-container:hover {
    box-shadow: var(--card-shadow-hover);
    transform: translateY(-2px);
}

/* Statistics Cards */
.stat-card {
    background: white;
    padding: var(--spacing-lg);
    border-radius: var(--radius-lg);
    text-align: center;
    box-shadow: var(--card-shadow);
    border: 1px solid rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--primary-gradient);
}

.stat-card:hover {
    box-shadow: var(--card-shadow-hover);
    transform: translateY(-4px);
}

.stat-value {
    font-size: 3rem;
    font-weight: 800;
    margin: var(--spacing-xs) 0;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 0.875rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    color: #2c3e50;
}

/* Modern Expanders */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: var(--radius-md);
    font-weight: 600;
    padding: var(--spacing-sm) var(--spacing-md);
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Premium Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f8f9fa;
    padding: 8px;
    border-radius: var(--radius-lg);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #495057;
    border-radius: var(--radius-md);
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-gradient) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* Animation for loading */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

/* Big font class */
.big-font {
    font-size: 20px !important;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Feature cards */
.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateX(10px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

/* Warning/Error boxes */
.stAlert {
    border-radius: 10px;
    border-left: 5px solid;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
</style>
""",
    unsafe_allow_html=True,
)


# Helper function to sanitize data for Excel export
def sanitize_for_excel(df):
    """Remove illegal characters from dataframe for Excel export."""
    import re

    # Excel illegal characters (control characters except tab, newline, carriage return)
    # CHAR(1) through CHAR(31) except CHAR(9), CHAR(10), CHAR(13)
    illegal_chars_pattern = r"[\x00-\x08\x0B\x0C\x0E-\x1F]"

    df_clean = df.copy()

    # Clean string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            try:
                df_clean[col] = df_clean[col].apply(
                    lambda x: (
                        re.sub(illegal_chars_pattern, "", str(x)) if pd.notna(x) else x
                    )
                )
            except:
                pass

    return df_clean


# Helper function to recommend ML vs Deep Learning
def get_ml_vs_dl_recommendation(
    df, target_column=None, problem_type=None, data_type="tabular"
):
    """
    Analyze dataset characteristics and recommend ML vs Deep Learning.

    Returns:
        dict: {
            'recommendation': 'ml' or 'dl',
            'confidence': 'high', 'medium', or 'low',
            'score': float (0-100, higher = more DL appropriate),
            'reasons': list of reasoning strings,
            'details': dict of metrics
        }
    """
    n_samples = len(df)
    n_features = len(df.columns) - (1 if target_column else 0)

    score = 0  # 0 = ML, 100 = DL
    reasons = []
    details = {}

    # Data type analysis (strongest signal)
    if data_type == "image":
        score += 80
        reasons.append("🖼️ Image data - Deep Learning excels at computer vision tasks")
        details["data_type"] = "Image (DL required)"
    elif data_type == "text":
        score += 75
        reasons.append(
            "📝 Text data - Deep Learning with embeddings handles NLP better"
        )
        details["data_type"] = "Text (DL recommended)"
    else:
        # Check for text-like columns in tabular data
        text_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype == "object":
                avg_length = df[col].astype(str).str.len().mean()
                unique_ratio = df[col].nunique() / len(df)
                # Text detection: long strings OR high cardinality (>50% unique)
                if avg_length > 50 or unique_ratio > 0.5:
                    text_columns.append(col)

        if text_columns:
            # Strong signal for DL - text data needs NLP models
            score += 75
            reasons.append(
                f"📄 Found {len(text_columns)} text column(s) - Deep Learning with NLP/embeddings required"
            )
            details["text_columns"] = len(text_columns)
            details["data_type"] = "Tabular with text (DL recommended)"
        else:
            reasons.append(
                "📊 Structured tabular data - Traditional ML is efficient and interpretable"
            )
            details["data_type"] = "Tabular (ML suitable)"

    # Dataset size analysis
    details["n_samples"] = n_samples
    details["n_features"] = n_features

    # Don't penalize DL for medium datasets if we have text columns
    has_text = details.get("text_columns", 0) > 0
    
    if n_samples < 1000:
        if not has_text:
            score -= 30
            reasons.append(
                f"📉 Small dataset ({n_samples:,} samples) - ML works better with limited data"
            )
        else:
            score -= 10
            reasons.append(
                f"📉 Small dataset ({n_samples:,} samples) - but text data still needs NLP"
            )
    elif n_samples < 10000:
        if not has_text:
            score -= 10
            reasons.append(
                f"📊 Medium dataset ({n_samples:,} samples) - ML is more sample-efficient"
            )
        else:
            # Don't penalize for text data
            reasons.append(
                f"📊 Medium dataset ({n_samples:,} samples) - sufficient for NLP models"
            )
    elif n_samples < 100000:
        score += 10
        reasons.append(
            f"📈 Large dataset ({n_samples:,} samples) - DL can leverage more data"
        )
    else:
        score += 30
        reasons.append(
            f"🚀 Very large dataset ({n_samples:,} samples) - DL scales better with data"
        )

    # Feature dimensionality
    if n_features < 10:
        score -= 20
        reasons.append(
            f"🔢 Low dimensionality ({n_features} features) - ML sufficient for simple problems"
        )
    elif n_features < 50:
        score -= 5
        reasons.append(
            f"📐 Medium dimensionality ({n_features} features) - ML handles this well"
        )
    elif n_features < 200:
        score += 5
        reasons.append(
            f"📊 High dimensionality ({n_features} features) - DL can learn complex patterns"
        )
    else:
        score += 15
        reasons.append(
            f"🎯 Very high dimensionality ({n_features} features) - DL excels at feature learning"
        )

    # Problem complexity (if target available)
    if target_column and target_column in df.columns:
        target = df[target_column]
        n_unique = target.nunique()

        if problem_type == "classification":
            if n_unique > 10:
                score += 10
                reasons.append(
                    f"🎨 Multi-class problem ({n_unique} classes) - DL handles many classes well"
                )
                details["n_classes"] = n_unique
            else:
                reasons.append(
                    f"🎯 Binary/simple classification ({n_unique} classes) - ML is fast and accurate"
                )
                details["n_classes"] = n_unique

    # Calculate recommendation
    score = max(0, min(100, score))  # Clamp to 0-100

    if score >= 60:
        recommendation = "dl"
        confidence = "high" if score >= 75 else "medium"
    elif score <= 40:
        recommendation = "ml"
        confidence = "high" if score <= 25 else "medium"
    else:
        recommendation = "ml"  # Default to ML in uncertain cases
        confidence = "low"

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "score": score,
        "reasons": reasons,
        "details": details,
    }


# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None
if "automl" not in st.session_state:
    st.session_state.automl = None
if "training_started" not in st.session_state:
    st.session_state.training_started = False
if "preprocessing_config" not in st.session_state:
    st.session_state.preprocessing_config = {}
if "data_type" not in st.session_state:
    st.session_state.data_type = "tabular"
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "text_data" not in st.session_state:
    st.session_state.text_data = None


def main():
    """Main application function."""

    # Professional Sidebar
    with st.sidebar:
        # Brand Header
        st.markdown(
            """
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;
                        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);'>
                <h1 style='color: white; margin: 0; font-size: 2rem; text-shadow: 0 2px 8px rgba(0,0,0,0.2);'>
                    🤖 AutoML<span style='opacity: 0.8;'> Pro</span>
                </h1>
                <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
                    Advanced Machine Learning Platform
                </p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='color: white; margin-top: 1rem;'>📍 Navigation</h3>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "📁 Data Upload",
                "📊 EDA Dashboard",
                "🎯 Train Models",
                "📈 Results & Comparison",
                "🚀 Quick Start",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        st.markdown("### About")
        st.info("""
        **AutoML System v1.0.0** 🚀
        
        Automated machine learning pipeline with:
        - **27+ ML/DL Models** with intelligent selection
        - **GPU Acceleration** for XGBoost, LightGBM, CatBoost & PyTorch
        - **Smart Recommendations** (ML vs DL auto-detection)
        - Hyperparameter optimization (Optuna)
        - Experiment tracking (MLflow)
        - **Advanced Preprocessing:**
          - KNN & Iterative Imputation
          - Outlier Detection & Handling
          - Feature Engineering
          - Feature Selection
        - **Traditional ML (18 models):**
          - 📊 XGBoost*, LightGBM*, CatBoost* (*GPU)
          - Random Forest, Gradient Boosting, SVM, KNN
          - Ridge, Lasso, ElasticNet, Naive Bayes
        - **Deep Learning (PyTorch):**
          - 📊 Tabular: MLP, ResNet, Attention, Wide & Deep
          - 🖼️ Vision: 9 architectures (ResNet, VGG, DenseNet, etc.)
          - 📝 NLP: LSTM/GRU/TextCNN/Attention
        - Interactive visualizations
        - Model export & download
        """)

        # Performance tips
        with st.expander("\u26a1 Performance Tips"):
            st.markdown("""
            **For Large Datasets:**
            
            1. **Limit Rows**: Use advanced loading options to load only a subset
            
            2. **Sampling**: System automatically uses sampling for >100k rows
            
            3. **Parquet Format**: Faster than CSV for large files
            
            4. **Memory**: Monitor the memory metric in data preview
            
            5. **Features**: Limit to most important features (<50 columns recommended)
            
            6. **Downloads**: For large datasets, click generate button first
            
            7. **Visualizations**: Based on samples, not full dataset
            """)

    # Main content based on selected page
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "📊 EDA Dashboard":
        eda_dashboard_page()
    elif page == "🎯 Train Models":
        train_models_page()
    elif page == "📈 Results & Comparison":
        results_page()
    elif page == "🚀 Quick Start":
        quick_start_page()


def data_upload_page():
    """Professional data upload and preview page."""

    # Page Header
    st.markdown(
        """
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5rem;'>📁 Data Upload & Preview</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem;'>
            Upload your dataset and get instant insights
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data Type Selection with Cards
    st.markdown(
        """
    <div style='margin: 2rem 0 1.5rem 0;'>
        <h2 style='font-size: 1.5rem; color: #2c3e50; margin-bottom: 1rem;'>
            🎯 Select Your Data Type
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Professional data type selector
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "📊 **Tabular Data**\n\nCSV, Excel, Parquet files", width="stretch"
        ):
            st.session_state.data_type = "tabular"

    with col2:
        if st.button("🖼️ **Image Data**\n\nPNG, JPG, image folders", width="stretch"):
            st.session_state.data_type = "image"

    with col3:
        if st.button("📝 **Text Data**\n\nText classification, NLP", width="stretch"):
            st.session_state.data_type = "text"

    # Show selected type indicator
    data_type_map = {"tabular": "📊 Tabular", "image": "🖼️ Image", "text": "📝 Text"}
    current_type = st.session_state.get("data_type", "tabular")
    st.markdown(
        f"""
    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                padding: 1rem; border-radius: 12px; text-align: center; margin: 1.5rem 0;'>
        <p style='margin: 0; font-weight: 600; color: #667eea;'>
            Selected: {data_type_map.get(current_type, 'None')}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Handle data type routing
    if current_type == "tabular":
        st.session_state.data_type = "tabular"
        tabular_data_upload()
    elif current_type == "image":
        st.session_state.data_type = "image"
        image_data_upload()
    elif current_type == "text":
        st.session_state.data_type = "text"
        text_data_upload()


def tabular_data_upload():
    """Tabular data upload interface."""

    # Upload Section Header
    st.markdown(
        """
    <div style='margin: 2rem 0 1rem 0;'>
        <h3 style='color: #2c3e50; font-size: 1.3rem;'>
            📤 Upload Your Dataset
        </h3>
        <p style='color: #7f8c8d; font-size: 0.95rem; margin: 0.5rem 0;'>
            Supported formats: <strong>CSV</strong>, <strong>Excel</strong>, <strong>Parquet</strong>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Performance options for large files
    with st.expander("⚙️ Advanced Loading Options", expanded=False):
        st.markdown("**For very large datasets:**")
        limit_rows = st.checkbox(
            "Limit number of rows",
            value=False,
            help="Only load a subset of rows for faster processing",
        )
        if limit_rows:
            max_rows = st.number_input(
                "Maximum rows to load",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=10000,
            )
        else:
            max_rows = None

    # File upload with custom styling
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "parquet"],
        help="Upload your dataset in CSV, Excel, or Parquet format",
    )

    if uploaded_file is not None:
        try:
            # Load data based on file type
            file_extension = Path(uploaded_file.name).suffix.lower()

            with st.spinner("Loading data..."):
                if file_extension == ".csv":
                    df = pd.read_csv(uploaded_file, nrows=max_rows)
                elif file_extension in [".xlsx", ".xls"]:
                    if max_rows:
                        df = pd.read_excel(uploaded_file, nrows=max_rows)
                    else:
                        df = pd.read_excel(uploaded_file)
                elif file_extension == ".parquet":
                    df = pd.read_parquet(uploaded_file)
                    if max_rows:
                        df = df.head(max_rows)
                else:
                    st.error(f"Unsupported file format: {file_extension}")
                    return

            if max_rows and len(df) == max_rows:
                st.info(
                    f"\u2139\ufe0f Loaded first {max_rows:,} rows. Original file may contain more data."
                )

            # Store in session state
            st.session_state.data = df

            # Check if dataset is large
            n_rows, n_cols = df.shape
            is_large = n_rows > 100000 or n_cols > 50

            # Detect text columns
            text_columns = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # Check if column contains long text strings
                    avg_length = df[col].astype(str).str.len().mean()
                    unique_ratio = df[col].nunique() / len(df)
                    if avg_length > 50 or unique_ratio > 0.5:  # Likely text data
                        text_columns.append(col)

            # Warn about text columns
            if text_columns:
                st.warning(f"⚠️ **Text columns detected:** {', '.join(text_columns)}")

                with st.expander("🔄 Convert to Text Data Format", expanded=True):
                    st.info("""
                    💡 **For text data (reviews, descriptions, etc.):**
                    - Use the **📝 Text Data** upload option instead
                    - This enables NLP models (LSTM, GRU, CNN, Attention)
                    - Traditional ML will struggle with high-cardinality text
                    """)

                    # Let user select text and label columns
                    st.markdown("**Quick Setup:**")

                    # Smart default: pick text column with longest avg length
                    text_col_default_idx = 0
                    if len(text_columns) > 1:
                        text_lengths = []
                        for col in text_columns:
                            avg_len = df[col].astype(str).str.len().mean()
                            text_lengths.append((col, avg_len))
                        text_col_default = max(text_lengths, key=lambda x: x[1])[0]
                        text_col_default_idx = text_columns.index(text_col_default)

                    # Smart default for label: column with fewest unique values
                    other_cols = [c for c in df.columns if c not in text_columns]
                    label_col_default_idx = 0
                    if other_cols and len(other_cols) > 1:
                        label_uniques = [(col, df[col].nunique()) for col in other_cols]
                        label_col_default = min(label_uniques, key=lambda x: x[1])[0]
                        try:
                            label_col_default_idx = other_cols.index(label_col_default)
                        except:
                            pass

                    col_setup1, col_setup2 = st.columns(2)

                    with col_setup1:
                        text_col_select = st.selectbox(
                            "Text Column (contains text to classify)",
                            options=text_columns,
                            index=text_col_default_idx,
                            key="quick_text_col",
                            help="Column with the actual text content (reviews, descriptions, etc.)",
                        )

                    with col_setup2:
                        label_col_select = st.selectbox(
                            "Label Column (target to predict)",
                            options=other_cols,
                            index=label_col_default_idx,
                            key="quick_label_col",
                            help="Column with categories/labels (sentiment, topic, etc.)",
                        )

                    if st.button("✅ Convert to Text Data Format", type="primary"):
                        # Store as text data
                        st.session_state.data_type = "text"
                        st.session_state.text_data = {
                            "dataframe": df,
                            "text_column": text_col_select,
                            "label_column": label_col_select,
                            "is_large": len(df) > 10000,
                        }
                        st.success(
                            f"✅ Converted! Text column: '{text_col_select}', Label column: '{label_col_select}'"
                        )
                        st.info(
                            "Now go to 🎯 Train Models and select 🔶 Deep Learning > NLP models"
                        )

                        # Show preview
                        st.markdown("**Preview:**")
                        preview_df = pd.DataFrame(
                            {
                                "Text": df[text_col_select].head(3),
                                "Label": df[label_col_select].head(3),
                            }
                        )
                        st.dataframe(preview_df)

            # Display success message
            if is_large:
                st.success(
                    f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns (Large dataset - using optimized display)"
                )
            else:
                st.success(
                    f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns"
                )

            # Data preview
            st.markdown("### 📋 Data Overview")

            # Enhanced metric cards with gradients
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    """
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{:,}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>ROWS</div>
                </div>
                """.format(len(df)),
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    """
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>COLUMNS</div>
                </div>
                """.format(len(df.columns)),
                    unsafe_allow_html=True,
                )

            with col3:
                # Use sampling for large datasets
                if is_large:
                    sample_df = df.sample(min(10000, len(df)), random_state=42)
                    missing_pct = (
                        sample_df.isnull().sum().sum()
                        / (len(sample_df) * len(sample_df.columns))
                        * 100
                    )
                else:
                    missing_pct = (
                        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                    )

                st.markdown(
                    """
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{:.1f}%</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>MISSING</div>
                </div>
                """.format(missing_pct),
                    unsafe_allow_html=True,
                )

            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.markdown(
                    """
                <div class='metric-card'>
                    <div style='font-size: 2.5rem; font-weight: bold;'>{:.1f}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>MB</div>
                </div>
                """.format(memory_mb),
                    unsafe_allow_html=True,
                )

            if is_large:
                st.caption("*Statistics computed on 10,000 row sample for performance")

            # Column Type Distribution Chart
            st.markdown("#### 📊 Column Type Distribution")
            type_counts = df.dtypes.astype(str).value_counts()

            col_chart1, col_chart2 = st.columns([1, 1])

            with col_chart1:
                # Pie chart for column types
                fig_types = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Data Types Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4,
                )
                fig_types.update_layout(
                    height=300,
                    showlegend=True,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_types, width="stretch")

            with col_chart2:
                # Numerical vs Categorical breakdown
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                breakdown_df = pd.DataFrame(
                    {
                        "Type": ["Numerical", "Categorical", "Other"],
                        "Count": [
                            len(numerical_cols),
                            len(categorical_cols),
                            len(df.columns)
                            - len(numerical_cols)
                            - len(categorical_cols),
                        ],
                    }
                )

                fig_breakdown = px.bar(
                    breakdown_df,
                    x="Type",
                    y="Count",
                    title="Feature Type Breakdown",
                    color="Type",
                    color_discrete_map={
                        "Numerical": "#667eea",
                        "Categorical": "#f093fb",
                        "Other": "#f5576c",
                    },
                )
                fig_breakdown.update_layout(
                    height=300,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_breakdown, width="stretch")

            # First rows with pagination
            st.markdown("#### Data Preview")

            # Pagination controls
            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            with col_p1:
                page_size = st.selectbox(
                    "Rows per page", [10, 25, 50, 100], index=0, key="preview_page_size"
                )
            with col_p2:
                max_pages = (len(df) - 1) // page_size + 1
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max_pages,
                    value=1,
                    key="preview_page",
                )
            with col_p3:
                st.metric("Total Pages", max_pages)

            # Display paginated data
            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            st.dataframe(df.iloc[start_idx:end_idx], width="stretch", height=400)

            # Data types
            st.markdown("#### Column Information")
            with st.spinner("Computing column statistics..."):
                # Use sampling for large datasets
                if is_large:
                    sample_df = df.sample(min(10000, len(df)), random_state=42)
                    col_types = pd.DataFrame(
                        {
                            "Column": df.columns,
                            "Type": df.dtypes.astype(str),
                            "Non-Null": sample_df.count() * (len(df) / len(sample_df)),
                            "Null": sample_df.isnull().sum()
                            * (len(df) / len(sample_df)),
                            "Unique": sample_df.nunique(),
                        }
                    ).reset_index(drop=True)
                    # Fix dtypes for display
                    col_types["Non-Null"] = col_types["Non-Null"].astype(int)
                    col_types["Null"] = col_types["Null"].astype(int)
                    st.caption("†Statistics estimated from 10,000 row sample")
                else:
                    col_types = pd.DataFrame(
                        {
                            "Column": df.columns,
                            "Type": df.dtypes.astype(str),
                            "Non-Null": df.count(),
                            "Null": df.isnull().sum(),
                            "Unique": df.nunique(),
                        }
                    ).reset_index(drop=True)
            st.dataframe(col_types, width="stretch", height=400)

            # Download processed data
            st.markdown("#### Export Data")

            # Only generate downloads on button click for large datasets
            if is_large:
                st.info(
                    "💡 For large datasets, click buttons below to generate download files"
                )
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📥 Generate CSV Download", key="gen_csv"):
                        with st.spinner("Generating CSV file..."):
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv,
                                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_csv",
                            )

                with col2:
                    if st.button("📥 Generate Excel Download", key="gen_excel"):
                        with st.spinner("Generating Excel file..."):
                            try:
                                buffer = io.BytesIO()
                                # Sanitize data for Excel
                                df_clean = sanitize_for_excel(df)
                                df_clean.to_excel(
                                    buffer, index=False, engine="openpyxl"
                                )
                                st.download_button(
                                    label="⬇️ Download Excel",
                                    data=buffer.getvalue(),
                                    file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel",
                                )
                            except Exception as excel_error:
                                st.error(
                                    f"⚠️ Excel export failed: {str(excel_error)[:100]}"
                                )
                                st.warning(
                                    "💡 Your data contains characters that Excel cannot handle. Please use CSV download instead."
                                )
                                # Offer CSV as fallback
                                csv = df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="📥 Download as CSV (Recommended)",
                                    data=csv,
                                    file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="download_csv_fallback",
                                )
            else:
                # For small datasets, generate immediately
                col1, col2 = st.columns(2)

                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                with col2:
                    try:
                        buffer = io.BytesIO()
                        # Sanitize data for Excel
                        df_clean = sanitize_for_excel(df)
                        df_clean.to_excel(buffer, index=False, engine="openpyxl")
                        st.download_button(
                            label="📥 Download as Excel",
                            data=buffer.getvalue(),
                            file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except Exception as excel_error:
                        st.warning(
                            "⚠️ Excel export not available for this dataset (contains unsupported characters)"
                        )
                        st.info(
                            "💡 Use CSV download instead - it supports all data types."
                        )

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)

    else:
        # Show example datasets
        st.markdown("### 📚 Or try an example dataset")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎯 Classification Example (Iris)", width="stretch"):
                from sklearn.datasets import load_iris

                iris = load_iris(as_frame=True)
                df = iris.frame  # type: ignore
                st.session_state.data = df
                st.rerun()

        with col2:
            if st.button("📈 Regression Example (Housing)", width="stretch"):
                from sklearn.datasets import fetch_california_housing

                housing = fetch_california_housing(as_frame=True)
                df = housing.frame  # type: ignore
                st.session_state.data = df
                st.rerun()


def image_data_upload():
    """Image data upload interface."""
    st.markdown("""
    Upload images for classification. You can upload a ZIP file, specify a folder path,
    or upload individual images and specify labels.
    """)

    upload_mode = st.radio(
        "Upload Mode:",
        [
            "📁 ZIP File (Folders as Classes)",
            "📂 Local Folder Path",
            "🖼️ Individual Images",
        ],
        help="Choose how you want to upload your images",
    )

    if "ZIP" in upload_mode:
        st.info("""📦 **ZIP Structure:**
        ```
        dataset.zip
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        ```
        """)

        uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])

        if uploaded_zip:
            import tempfile
            import zipfile

            # Show file info
            zip_size_mb = uploaded_zip.size / (1024 * 1024)
            st.info(f"📦 ZIP size: {zip_size_mb:.1f} MB")

            with st.spinner("Extracting and scanning for images..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Scan for images
                    image_files = []
                    labels = []

                    from pathlib import Path

                    temp_path = Path(temp_dir)

                    for class_dir in temp_path.iterdir():
                        if class_dir.is_dir():
                            class_name = class_dir.name
                            for img_file in class_dir.glob("*"):
                                if img_file.suffix.lower() in [
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                    ".bmp",
                                ]:
                                    image_files.append(str(img_file))
                                    labels.append(class_name)

                if image_files:
                    st.session_state.image_data = {
                        "files": image_files,
                        "labels": labels,
                    }
                    st.success(
                        f"✅ Loaded {len(image_files)} images from {len(set(labels))} classes"
                    )

                    # Show class distribution
                    import pandas as pd

                    class_counts = pd.Series(labels).value_counts()
                    st.markdown("**Class Distribution:**")
                    st.bar_chart(class_counts)
                else:
                    st.warning("No images found in ZIP file")

    elif "Folder" in upload_mode:
        st.info("""📂 **Folder Structure:**
        ```
        your_folder/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        ```
        Each subfolder name will be used as the class label.
        """)

        # Initialize session state for folder path
        if "selected_folder_path" not in st.session_state:
            st.session_state.selected_folder_path = ""

        # Create drag-and-drop style interface
        st.markdown(
            """
        <style>
        .folder-upload-container {
            background-color: #262730;
            border: 2px dashed #4a4a4a;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            position: relative;
        }
        .folder-upload-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            opacity: 0.6;
        }
        .folder-upload-text {
            color: #ffffff;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .folder-upload-subtext {
            color: #a0a0a0;
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }
        .browse-btn-container {
            position: absolute;
            top: 2rem;
            right: 2rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display selected path or prompt
        if st.session_state.selected_folder_path:
            display_text = st.session_state.selected_folder_path
            subtext = "Folder selected • Click Browse to change"
        else:
            display_text = "Select folder containing image classes"
            subtext = "Organize images in subfolders by class"

        # Create the upload box with Browse button
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(
                f"""
            <div class='folder-upload-container'>
                <div class='folder-upload-icon'>📁</div>
                <div class='folder-upload-text'>{display_text}</div>
                <div class='folder-upload-subtext'>{subtext}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Browse files", width="stretch", type="secondary"):
                try:
                    import tkinter as tk
                    from tkinter import filedialog

                    # Create a Tk root window (hidden)
                    root = tk.Tk()
                    root.withdraw()
                    root.wm_attributes("-topmost", 1)

                    # Open folder selection dialog
                    selected_folder = filedialog.askdirectory(
                        title="Select Image Folder",
                        initialdir=(
                            st.session_state.selected_folder_path
                            if st.session_state.selected_folder_path
                            else None
                        ),
                    )

                    # Clean up
                    root.destroy()

                    if selected_folder:
                        st.session_state.selected_folder_path = selected_folder
                        st.rerun()
                except ImportError:
                    st.error(
                        "❌ Tkinter not available. Please enter the path manually."
                    )
                except Exception as e:
                    st.error(f"❌ Error opening folder browser: {str(e)}")

        folder_path = st.session_state.selected_folder_path

        if folder_path:
            import os
            from pathlib import Path

            folder_path_obj = Path(folder_path)

            if not folder_path_obj.exists():
                st.error(f"❌ Folder does not exist: {folder_path}")
            elif not folder_path_obj.is_dir():
                st.error(f"❌ Path is not a folder: {folder_path}")
            else:
                with st.spinner("Scanning folder for images..."):
                    # Scan for images
                    image_files = []
                    labels = []

                    # Recursively find all class folders
                    class_folders = [d for d in folder_path_obj.iterdir() if d.is_dir()]

                    if not class_folders:
                        st.warning(
                            "⚠️ No subfolders found. Please organize images into class folders."
                        )
                    else:
                        for class_dir in class_folders:
                            class_name = class_dir.name
                            # Skip hidden folders and common non-class folders
                            if class_name.startswith(".") or class_name.lower() in [
                                "__pycache__",
                                "checkpoints",
                            ]:
                                continue

                            for img_file in class_dir.rglob("*"):
                                if img_file.is_file() and img_file.suffix.lower() in [
                                    ".jpg",
                                    ".jpeg",
                                    ".png",
                                    ".bmp",
                                    ".gif",
                                    ".webp",
                                ]:
                                    image_files.append(str(img_file))
                                    labels.append(class_name)

                    if image_files:
                        st.session_state.image_data = {
                            "files": image_files,
                            "labels": labels,
                        }
                        st.success(
                            f"✅ Loaded {len(image_files)} images from {len(set(labels))} classes"
                        )

                        # Show class distribution
                        import pandas as pd

                        class_counts = pd.Series(labels).value_counts()
                        st.markdown("**Class Distribution:**")
                        st.bar_chart(class_counts)

                        # Show preview of classes found
                        st.markdown("**Classes found:**")
                        st.write(", ".join(sorted(set(labels))))
                    else:
                        st.warning(
                            "No images found in folder. Make sure images are in class subfolders."
                        )

    else:
        st.info("Upload images and specify their labels manually")
        uploaded_images = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_images:
            st.write(f"Uploaded {len(uploaded_images)} images")
            # For demo purposes - in production, you'd collect labels
            st.warning(
                "⚠️ Manual labeling interface coming soon. Use ZIP upload for now."
            )


def text_data_upload():
    """Text data upload interface."""
    st.markdown("""
    Upload text data for classification (sentiment analysis, topic classification, etc.)
    """)

    # Performance options for large text datasets
    with st.expander("⚙️ Advanced Loading Options"):
        st.markdown("**For very large text datasets:**")
        limit_rows_text = st.checkbox(
            "Limit number of rows",
            value=False,
            help="Only load a subset of rows for faster processing",
            key="limit_text_rows",
        )
        if limit_rows_text:
            max_rows_text = st.number_input(
                "Maximum rows to load",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                key="max_text_rows",
            )
        else:
            max_rows_text = None

    upload_mode = st.radio(
        "Upload Mode:",
        ["📄 CSV File (Text + Labels)", "✍️ Enter Text Manually"],
        help="Choose how you want to provide your text data",
    )

    if "CSV" in upload_mode:
        st.info("""📊 **CSV Format:**
        Your CSV should have at least two columns:
        - One column for the text
        - One column for the label/class
        """)

        uploaded_file = st.file_uploader(
            "Upload CSV file", type=["csv"], help="CSV with text and labels"
        )

        if uploaded_file:
            import pandas as pd

            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📄 File size: {file_size_mb:.1f} MB")

            # Add progress indicator for large files
            if file_size_mb > 10:
                st.warning("⏳ Large file detected. Loading may take a moment...")

            with st.spinner("Loading text data..."):
                # Load with row limit if specified
                if max_rows_text:
                    df = pd.read_csv(uploaded_file, nrows=max_rows_text)
                else:
                    df = pd.read_csv(uploaded_file)

            st.session_state.data = df

            # Check if dataset is large
            n_rows = len(df)
            is_large_text = n_rows > 10000

            if max_rows_text and n_rows == max_rows_text:
                st.info(
                    f"ℹ️ Loaded first {max_rows_text:,} rows. Original file may contain more data."
                )

            st.success(f"✅ Loaded {len(df):,} text samples")

            # Show preview with pagination
            st.markdown("**Data Preview:**")

            # Pagination controls
            col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
            with col_p1:
                page_size_text = st.selectbox(
                    "Rows per page",
                    [5, 10, 25, 50],
                    index=1,
                    key="text_preview_page_size",
                )
            with col_p2:
                max_pages_text = (len(df) - 1) // page_size_text + 1
                current_page_text = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max_pages_text,
                    value=1,
                    key="text_preview_page",
                )
            with col_p3:
                st.metric("Total Pages", max_pages_text)

            # Display paginated data
            start_idx_text = (current_page_text - 1) * page_size_text
            end_idx_text = min(start_idx_text + page_size_text, len(df))
            st.dataframe(
                df.iloc[start_idx_text:end_idx_text], width="stretch", height=300
            )

            # Column selection with auto-detection
            st.markdown("**Select Columns:**")

            # Auto-detect text and label columns
            text_col_default = 0
            label_col_default = 1 if len(df.columns) > 1 else 0

            # Heuristic: Text column typically has longer average length
            # Label column typically has fewer unique values
            if len(df.columns) >= 2:
                col_stats = []
                for col in df.columns:
                    try:
                        avg_len = df[col].astype(str).str.len().mean()
                        unique_ratio = df[col].nunique() / len(df)
                        col_stats.append(
                            {
                                "column": col,
                                "avg_length": avg_len,
                                "unique_ratio": unique_ratio,
                                "nunique": df[col].nunique(),
                            }
                        )
                    except:
                        col_stats.append(
                            {
                                "column": col,
                                "avg_length": 0,
                                "unique_ratio": 0,
                                "nunique": 0,
                            }
                        )

                # Sort by average length descending
                col_stats_sorted = sorted(
                    col_stats, key=lambda x: x["avg_length"], reverse=True
                )

                # Text column: longest average length
                detected_text_col = col_stats_sorted[0]["column"]

                # Label column: among remaining columns, pick one with lowest unique ratio
                remaining_stats = [
                    s for s in col_stats_sorted if s["column"] != detected_text_col
                ]
                if remaining_stats:
                    detected_label_col = min(
                        remaining_stats, key=lambda x: x["unique_ratio"]
                    )["column"]
                else:
                    detected_label_col = (
                        col_stats_sorted[1]["column"]
                        if len(col_stats_sorted) > 1
                        else detected_text_col
                    )

                # Get indices for defaults
                try:
                    text_col_default = df.columns.tolist().index(detected_text_col)
                    label_col_default = df.columns.tolist().index(detected_label_col)
                except:
                    pass

                st.info(
                    f"💡 Auto-detected: Text='{detected_text_col}' (avg {col_stats_sorted[0]['avg_length']:.0f} chars), Label='{detected_label_col}' ({df[detected_label_col].nunique()} unique values)"
                )

            col1, col2 = st.columns(2)

            with col1:
                text_column = st.selectbox(
                    "Text Column (input features)",
                    df.columns.tolist(),
                    index=text_col_default,
                    help="Column containing the text to classify",
                )

            with col2:
                label_column = st.selectbox(
                    "Label Column (target to predict)",
                    df.columns.tolist(),
                    index=label_col_default,
                    help="Column containing the labels/categories",
                )

            if text_column and label_column:
                # For large datasets, store reference to dataframe instead of converting to list
                if is_large_text:
                    st.info(
                        "💡 Large dataset detected. Data will be processed in batches during training."
                    )

                    st.session_state.text_data = {
                        "dataframe": df,
                        "text_column": text_column,
                        "label_column": label_column,
                        "is_large": True,
                    }
                else:
                    st.session_state.text_data = {
                        "texts": df[text_column].tolist(),
                        "labels": df[label_column].tolist(),
                        "text_column": text_column,
                        "label_column": label_column,
                        "is_large": False,
                    }

                # Show class distribution with sampling for large datasets
                st.markdown("**Class Distribution:**")
                with st.spinner("Computing class distribution..."):
                    if is_large_text:
                        # Use sample for large datasets
                        sample_df = df.sample(min(10000, len(df)), random_state=42)
                        class_counts = sample_df[label_column].value_counts().head(20)
                        st.caption("†Based on 10,000 sample (showing top 20 classes)")
                    else:
                        class_counts = df[label_column].value_counts()
                st.bar_chart(class_counts)

    else:
        st.info("Enter text samples manually (useful for quick testing)")

        num_samples = st.number_input(
            "Number of samples", min_value=2, max_value=20, value=5
        )

        texts = []
        labels = []

        for i in range(num_samples):
            col1, col2 = st.columns([3, 1])
            with col1:
                text = st.text_area(f"Sample {i+1}", key=f"text_{i}", height=80)
                texts.append(text)
            with col2:
                label = st.text_input(f"Label {i+1}", key=f"label_{i}")
                labels.append(label)

        if st.button("Save Manual Data"):
            # Filter out empty entries
            valid_data = [
                (t, l) for t, l in zip(texts, labels) if t.strip() and l.strip()
            ]

            if valid_data:
                st.session_state.text_data = {
                    "texts": [t for t, l in valid_data],
                    "labels": [l for t, l in valid_data],
                }
                st.success(f"✅ Saved {len(valid_data)} text samples")
            else:
                st.warning("Please enter at least some text and labels")


def eda_dashboard_page():
    """Exploratory data analysis dashboard."""

    # Professional Header
    st.markdown(
        """
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5rem;'>📊 Exploratory Data Analysis</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem;'>
            Discover insights and patterns in your data
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if any data type is loaded
    has_tabular_data = st.session_state.data is not None
    has_image_data = st.session_state.get("image_data") is not None
    has_text_data = st.session_state.get("text_data") is not None

    if not (has_tabular_data or has_image_data or has_text_data):
        st.warning("⚠️ Please upload data first!")
        return

    # Handle different data types
    if has_image_data and not has_tabular_data:
        # Image Data EDA
        image_data = st.session_state.image_data
        files = image_data["files"]
        labels = image_data["labels"]

        st.markdown(
            """
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
            <h3 style='margin: 0; color: #667eea;'>📸 Image Dataset Analysis</h3>
            <p style='margin: 0.5rem 0 0 0; color: #7f8c8d;'>
                Analyze your image dataset structure and characteristics
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Tabs for image EDA
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Overview",
                "🏷️ Class Distribution",
                "🖼️ Sample Images",
                "📐 Image Properties",
            ]
        )

        with tab1:
            st.markdown("### Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", f"{len(files):,}")
            with col2:
                num_classes = len(set(labels))
                st.metric("Classes", num_classes)
            with col3:
                avg_per_class = len(files) / num_classes
                st.metric("Avg/Class", f"{avg_per_class:.0f}")
            with col4:
                # Check class balance
                from collections import Counter

                label_counts = Counter(labels)
                max_count = max(label_counts.values())
                min_count = min(label_counts.values())
                balance_ratio = min_count / max_count if max_count > 0 else 0
                balance_status = (
                    "✅ Balanced" if balance_ratio > 0.7 else "⚠️ Imbalanced"
                )
                st.metric("Balance", balance_status)

            # Class breakdown
            st.markdown("### Class Breakdown")
            class_df = pd.DataFrame(
                [
                    {
                        "Class": cls,
                        "Count": count,
                        "Percentage": f"{count/len(files)*100:.1f}%",
                    }
                    for cls, count in sorted(
                        label_counts.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )
            st.dataframe(class_df, width="stretch", hide_index=True)

        with tab2:
            st.markdown("### Class Distribution")

            from collections import Counter

            label_counts = Counter(labels)

            # Distribution bar chart
            dist_df = pd.DataFrame(
                [
                    {"Class": cls, "Count": count}
                    for cls, count in sorted(
                        label_counts.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )

            fig = px.bar(
                dist_df,
                x="Class",
                y="Count",
                title="Images per Class",
                color="Count",
                color_continuous_scale=["#667eea", "#764ba2"],
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=False,
                xaxis=dict(title="Class", tickangle=-45),
                yaxis=dict(title="Number of Images"),
                title=dict(x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig, width="stretch")

            # Pie chart
            fig_pie = px.pie(
                dist_df,
                values="Count",
                names="Class",
                title="Class Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )

            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                title=dict(x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig_pie, width="stretch")

        with tab3:
            st.markdown("### Sample Images")

            # Show samples from each class
            num_samples = st.slider("Samples per class", 3, 10, 5)

            for class_name in sorted(set(labels)):
                st.markdown(f"#### {class_name}")
                class_files = [f for f, l in zip(files, labels) if l == class_name]

                # Select random samples
                import random

                sample_files = random.sample(
                    class_files, min(num_samples, len(class_files))
                )

                # Display in columns
                cols = st.columns(min(num_samples, len(sample_files)))
                for idx, img_path in enumerate(sample_files):
                    with cols[idx]:
                        try:
                            from PIL import Image

                            img = Image.open(img_path)
                            st.image(
                                img,
                                width="stretch",
                                caption=f"{img.size[0]}x{img.size[1]}",
                            )
                        except:
                            st.error(f"Error loading {img_path}")

        with tab4:
            st.markdown("### Image Properties Analysis")

            with st.spinner("Analyzing image dimensions..."):
                from PIL import Image

                # Sample images for analysis (limit for performance)
                sample_size = min(500, len(files))
                sample_indices = random.sample(range(len(files)), sample_size)
                sample_files = [files[i] for i in sample_indices]

                dimensions = []
                file_sizes = []
                formats = []

                for img_path in sample_files:
                    try:
                        img = Image.open(img_path)
                        dimensions.append((img.size[0], img.size[1]))
                        file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
                        formats.append(img.format)
                    except:
                        pass

                if dimensions:
                    widths = [d[0] for d in dimensions]
                    heights = [d[1] for d in dimensions]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Dimension Statistics")
                        dim_stats = pd.DataFrame(
                            {
                                "Metric": [
                                    "Min Width",
                                    "Max Width",
                                    "Avg Width",
                                    "Min Height",
                                    "Max Height",
                                    "Avg Height",
                                ],
                                "Value": [
                                    f"{min(widths)}px",
                                    f"{max(widths)}px",
                                    f"{sum(widths)/len(widths):.0f}px",
                                    f"{min(heights)}px",
                                    f"{max(heights)}px",
                                    f"{sum(heights)/len(heights):.0f}px",
                                ],
                            }
                        )
                        st.dataframe(dim_stats, hide_index=True, width="stretch")

                        # File size stats
                        st.markdown("#### File Size Statistics")
                        st.metric(
                            "Average Size", f"{sum(file_sizes)/len(file_sizes):.1f} KB"
                        )
                        st.metric(
                            "Total Size (estimated)",
                            f"{sum(file_sizes) * len(files) / sample_size / 1024:.1f} MB",
                        )

                    with col2:
                        # Dimension distribution
                        fig = px.scatter(
                            x=widths,
                            y=heights,
                            title="Image Dimensions Distribution",
                            labels={"x": "Width (px)", "y": "Height (px)"},
                            opacity=0.6,
                        )

                        fig.update_traces(marker=dict(color="#667eea", size=8))
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=400,
                            title=dict(x=0.5, xanchor="center"),
                        )

                        st.plotly_chart(fig, width="stretch")

                        # Format distribution
                        from collections import Counter

                        format_counts = Counter(formats)
                        st.markdown("#### Image Formats")
                        for fmt, count in format_counts.most_common():
                            st.write(
                                f"**{fmt}**: {count} images ({count/len(formats)*100:.1f}%)"
                            )

                    if sample_size < len(files):
                        st.caption(f"†Analysis based on {sample_size} random samples")

        return

    elif has_text_data and not has_tabular_data:
        # Text Data EDA
        text_data = st.session_state.text_data

        # Extract texts and labels
        if "dataframe" in text_data:
            df = text_data["dataframe"]
            texts = df[text_data["text_column"]].tolist()
            labels = df[text_data["label_column"]].tolist()
        else:
            texts = text_data["texts"]
            labels = text_data["labels"]

        st.markdown(
            """
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
            <h3 style='margin: 0; color: #667eea;'>📝 Text Dataset Analysis</h3>
            <p style='margin: 0.5rem 0 0 0; color: #7f8c8d;'>
                Analyze your text dataset structure and characteristics
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Tabs for text EDA
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Overview",
                "📏 Text Statistics",
                "🏷️ Class Distribution",
                "📄 Sample Texts",
            ]
        )

        with tab1:
            st.markdown("### Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Texts", f"{len(texts):,}")
            with col2:
                num_classes = len(set(labels))
                st.metric("Classes", num_classes)
            with col3:
                avg_per_class = len(texts) / num_classes
                st.metric("Avg/Class", f"{avg_per_class:.0f}")
            with col4:
                # Check class balance
                from collections import Counter

                label_counts = Counter(labels)
                max_count = max(label_counts.values())
                min_count = min(label_counts.values())
                balance_ratio = min_count / max_count if max_count > 0 else 0
                balance_status = (
                    "✅ Balanced" if balance_ratio > 0.7 else "⚠️ Imbalanced"
                )
                st.metric("Balance", balance_status)

            # Class breakdown
            st.markdown("### Class Breakdown")
            class_df = pd.DataFrame(
                [
                    {
                        "Class": cls,
                        "Count": count,
                        "Percentage": f"{count/len(texts)*100:.1f}%",
                    }
                    for cls, count in sorted(
                        label_counts.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )
            st.dataframe(class_df, width="stretch", hide_index=True)

        with tab2:
            st.markdown("### Text Length Analysis")

            with st.spinner("Analyzing text characteristics..."):
                # Calculate statistics
                text_lengths = [len(str(t)) for t in texts]
                word_counts = [len(str(t).split()) for t in texts]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Character Count")
                    char_stats = pd.DataFrame(
                        {
                            "Metric": ["Min", "Max", "Mean", "Median"],
                            "Characters": [
                                f"{min(text_lengths):,}",
                                f"{max(text_lengths):,}",
                                f"{sum(text_lengths)/len(text_lengths):.0f}",
                                f"{sorted(text_lengths)[len(text_lengths)//2]:,}",
                            ],
                        }
                    )
                    st.dataframe(char_stats, hide_index=True, width="stretch")

                    # Character length distribution
                    fig_chars = px.histogram(
                        x=text_lengths,
                        nbins=50,
                        title="Character Count Distribution",
                        labels={"x": "Characters", "y": "Frequency"},
                        color_discrete_sequence=["#667eea"],
                    )

                    fig_chars.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=300,
                        showlegend=False,
                        title=dict(x=0.5, xanchor="center", font=dict(size=14)),
                    )

                    st.plotly_chart(fig_chars, width="stretch")

                with col2:
                    st.markdown("#### Word Count")
                    word_stats = pd.DataFrame(
                        {
                            "Metric": ["Min", "Max", "Mean", "Median"],
                            "Words": [
                                f"{min(word_counts)}",
                                f"{max(word_counts)}",
                                f"{sum(word_counts)/len(word_counts):.0f}",
                                f"{sorted(word_counts)[len(word_counts)//2]}",
                            ],
                        }
                    )
                    st.dataframe(word_stats, hide_index=True, width="stretch")

                    # Word count distribution
                    fig_words = px.histogram(
                        x=word_counts,
                        nbins=50,
                        title="Word Count Distribution",
                        labels={"x": "Words", "y": "Frequency"},
                        color_discrete_sequence=["#764ba2"],
                    )

                    fig_words.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=300,
                        showlegend=False,
                        title=dict(x=0.5, xanchor="center", font=dict(size=14)),
                    )

                    st.plotly_chart(fig_words, width="stretch")

            # Vocabulary analysis
            st.markdown("### Vocabulary Analysis")

            with st.spinner("Building vocabulary..."):
                # Sample for large datasets
                sample_size = min(5000, len(texts))
                import random as random_module

                sample_texts = random_module.sample(
                    [str(t) for t in texts], sample_size
                )

                all_words = []
                for text in sample_texts:
                    words = text.lower().split()
                    all_words.extend(words)

                from collections import Counter

                word_freq = Counter(all_words)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Unique Words (sample)", f"{len(word_freq):,}")
                    st.metric("Total Words (sample)", f"{len(all_words):,}")

                    # Top words
                    st.markdown("#### Most Common Words")
                    top_words = pd.DataFrame(
                        word_freq.most_common(15), columns=["Word", "Frequency"]
                    )
                    st.dataframe(top_words, hide_index=True, width="stretch")

                with col2:
                    # Word frequency chart
                    top_20 = word_freq.most_common(20)
                    fig_words = px.bar(
                        x=[w[0] for w in top_20],
                        y=[w[1] for w in top_20],
                        title="Top 20 Most Frequent Words",
                        labels={"x": "Word", "y": "Frequency"},
                        color=[w[1] for w in top_20],
                        color_continuous_scale="Blues",
                    )

                    fig_words.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=400,
                        showlegend=False,
                        xaxis=dict(tickangle=-45),
                        title=dict(x=0.5, xanchor="center"),
                    )

                    st.plotly_chart(fig_words, width="stretch")

                if sample_size < len(texts):
                    st.caption(
                        f"†Vocabulary analysis based on {sample_size} random samples"
                    )

        with tab3:
            st.markdown("### Class Distribution")

            from collections import Counter

            label_counts = Counter(labels)

            # Distribution bar chart
            dist_df = pd.DataFrame(
                [
                    {"Class": str(cls), "Count": count}
                    for cls, count in sorted(
                        label_counts.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )

            fig = px.bar(
                dist_df,
                x="Class",
                y="Count",
                title="Texts per Class",
                color="Count",
                color_continuous_scale=["#667eea", "#764ba2"],
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=False,
                xaxis=dict(title="Class", tickangle=-45),
                yaxis=dict(title="Number of Texts"),
                title=dict(x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig, width="stretch")

            # Text length by class
            st.markdown("### Average Text Length by Class")

            class_lengths = {}
            for text, label in zip(texts, labels):
                if label not in class_lengths:
                    class_lengths[label] = []
                class_lengths[label].append(len(str(text).split()))

            avg_lengths = pd.DataFrame(
                [
                    {"Class": str(cls), "Avg Words": sum(lengths) / len(lengths)}
                    for cls, lengths in class_lengths.items()
                ]
            ).sort_values("Avg Words", ascending=False)

            fig_avg = px.bar(
                avg_lengths,
                x="Class",
                y="Avg Words",
                title="Average Word Count by Class",
                color="Avg Words",
                color_continuous_scale="Purples",
            )

            fig_avg.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=False,
                xaxis=dict(tickangle=-45),
                title=dict(x=0.5, xanchor="center"),
            )

            st.plotly_chart(fig_avg, width="stretch")

        with tab4:
            st.markdown("### Sample Texts")

            # Show samples from each class
            num_samples = st.slider("Samples per class", 3, 10, 5)

            for class_name in sorted(set(labels)):
                st.markdown(f"#### Class: {class_name}")
                class_texts = [t for t, l in zip(texts, labels) if l == class_name]

                # Select random samples
                import random as random_module

                sample_texts = random_module.sample(
                    class_texts, min(num_samples, len(class_texts))
                )

                for idx, text in enumerate(sample_texts, 1):
                    text_str = str(text)
                    word_count = len(text_str.split())
                    char_count = len(text_str)

                    st.markdown(
                        f"""
                    <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #667eea;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #666;'>
                            Sample {idx} | {word_count} words | {char_count} chars
                        </p>
                        <p style='margin: 0.5rem 0 0 0; color: #333;'>
                            {text_str[:500]}{'...' if len(text_str) > 500 else ''}
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        return

    elif not has_tabular_data:
        st.warning("⚠️ No data loaded for EDA analysis!")
        return

    df = st.session_state.data
    n_rows, n_cols = df.shape
    is_large = n_rows > 100000 or n_cols > 50

    # Show dataset size warning with better styling
    if is_large:
        st.markdown(
            f"""
        <div style='background: linear-gradient(135deg, rgba(255, 165, 38, 0.1) 0%, rgba(255, 122, 0, 0.1) 100%);
                    padding: 1rem; border-radius: 12px; border-left: 4px solid #ffa726; margin: 1.5rem 0;'>
            <p style='margin: 0; font-weight: 600; color: #e65100;'>
                💡 Large dataset detected ({n_rows:,} rows, {n_cols} columns). Using optimized analysis with sampling for performance.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Tabs for different EDA sections
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Statistical Summary",
            "📊 Distributions",
            "🔗 Correlations",
            "🔍 Missing Values",
        ]
    )

    with tab1:
        st.markdown("### Statistical Summary")

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Numerical Features")
            if numerical_cols:
                with st.spinner("Computing statistics..."):
                    if is_large:
                        # Use sample for large datasets
                        sample_df = df[numerical_cols].sample(
                            min(10000, len(df)), random_state=42
                        )
                        st.dataframe(sample_df.describe(), width="stretch")
                        st.caption("†Statistics based on 10,000 row sample")
                    else:
                        st.dataframe(df[numerical_cols].describe(), width="stretch")
            else:
                st.info("No numerical features found")

        with col2:
            st.markdown("#### Categorical Features")
            if categorical_cols:
                with st.spinner("Analyzing categorical features..."):
                    # Limit categorical analysis for large datasets
                    analysis_cols = (
                        categorical_cols if not is_large else categorical_cols[:20]
                    )

                    if is_large:
                        sample_df = df[analysis_cols].sample(
                            min(10000, len(df)), random_state=42
                        )
                        cat_summary = pd.DataFrame(
                            {
                                "Column": analysis_cols,
                                "Unique Values": [
                                    sample_df[col].nunique() for col in analysis_cols
                                ],
                                "Most Frequent": [
                                    (
                                        sample_df[col].mode()[0]
                                        if len(sample_df[col].mode()) > 0
                                        else None
                                    )
                                    for col in analysis_cols
                                ],
                                "Frequency": [
                                    (
                                        sample_df[col].value_counts().iloc[0]
                                        if len(sample_df[col]) > 0
                                        else 0
                                    )
                                    for col in analysis_cols
                                ],
                            }
                        )
                        st.dataframe(cat_summary, width="stretch")
                        if len(categorical_cols) > 20:
                            st.caption(
                                f"†Showing first 20 of {len(categorical_cols)} categorical columns"
                            )
                        st.caption("†Statistics based on sample")
                    else:
                        cat_summary = pd.DataFrame(
                            {
                                "Column": categorical_cols,
                                "Unique Values": [
                                    df[col].nunique() for col in categorical_cols
                                ],
                                "Most Frequent": [
                                    (
                                        df[col].mode()[0]
                                        if len(df[col].mode()) > 0
                                        else None
                                    )
                                    for col in categorical_cols
                                ],
                                "Frequency": [
                                    (
                                        df[col].value_counts().iloc[0]
                                        if len(df[col]) > 0
                                        else 0
                                    )
                                    for col in categorical_cols
                                ],
                            }
                        )
                        st.dataframe(cat_summary, width="stretch")
            else:
                st.info("No categorical features found")

    with tab2:
        st.markdown("### Feature Distributions")

        if numerical_cols:
            selected_feature = st.selectbox(
                "Select feature to visualize", numerical_cols
            )

            col1, col2 = st.columns(2)

            # Use sampling for large datasets in visualizations
            plot_df = (
                df if not is_large else df.sample(min(10000, len(df)), random_state=42)
            )

            with col1:
                # Enhanced Histogram
                with st.spinner("Creating histogram..."):
                    fig_hist = px.histogram(
                        plot_df,
                        x=selected_feature,
                        title=f"Distribution of {selected_feature}",
                        marginal="box",
                        color_discrete_sequence=["#667eea"],
                    )

                    fig_hist.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        height=450,
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(200,200,200,0.2)",
                            title_font=dict(size=14, color="#666"),
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(200,200,200,0.2)",
                            title_font=dict(size=14, color="#666"),
                        ),
                        title=dict(
                            font=dict(size=16, color="#333"), x=0.5, xanchor="center"
                        ),
                    )

                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.plotly_chart(fig_hist, width="stretch")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if is_large:
                        st.caption("†Based on 10,000 row sample")

            with col2:
                # Enhanced Box plot
                with st.spinner("Creating box plot..."):
                    fig_box = px.box(
                        plot_df,
                        y=selected_feature,
                        title=f"Box Plot of {selected_feature}",
                        color_discrete_sequence=["#f5576c"],
                    )

                    fig_box.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        height=450,
                        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(200,200,200,0.2)",
                            title_font=dict(size=14, color="#666"),
                        ),
                        title=dict(
                            font=dict(size=16, color="#333"), x=0.5, xanchor="center"
                        ),
                    )

                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.plotly_chart(fig_box, width="stretch")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if is_large:
                        st.caption("†Based on 10,000 row sample")
        else:
            st.info("No numerical features to visualize")

    with tab3:
        st.markdown("### Correlation Analysis")

        if len(numerical_cols) > 1:
            with st.spinner("Computing correlations..."):
                # Limit features for correlation matrix on large datasets
                if len(numerical_cols) > 30:
                    st.info(
                        f"💡 Dataset has {len(numerical_cols)} numerical features. Showing top 30 by variance for readability."
                    )

                    # Select top N features by variance
                    variances = df[numerical_cols].var().sort_values(ascending=False)
                    top_features = variances.head(30).index.tolist()
                    corr_cols = top_features
                else:
                    corr_cols = numerical_cols

                # Use sampling for large datasets
                if is_large:
                    sample_df = df[corr_cols].sample(
                        min(10000, len(df)), random_state=42
                    )
                    corr_matrix = sample_df.corr()
                    st.caption("†Correlations computed on 10,000 row sample")
                else:
                    corr_matrix = df[corr_cols].corr()

            # Enhanced correlation heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale=[
                    [0, "#f5576c"],  # Strong negative - red
                    [0.5, "#f5f5f5"],  # Neutral - white
                    [1, "#667eea"],  # Strong positive - blue
                ],
                zmin=-1,
                zmax=1,
                title="Feature Correlation Heatmap",
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
                height=600,
                title=dict(font=dict(size=18, color="#333"), x=0.5, xanchor="center"),
                xaxis=dict(side="bottom", tickangle=-45),
                coloraxis=dict(
                    colorbar=dict(
                        title=dict(text="Correlation", font=dict(size=12)),
                        tickfont=dict(size=10),
                    )
                ),
            )

            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            # Top correlations
            st.markdown("#### Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_pairs.append(
                        {
                            "Feature 1": corr_matrix.columns[i],
                            "Feature 2": corr_matrix.columns[j],
                            "Correlation": corr_matrix.iloc[i, j],
                        }
                    )

            corr_df = (
                pd.DataFrame(corr_pairs)
                .sort_values("Correlation", key=abs, ascending=False)
                .head(10)
            )
            st.dataframe(corr_df, width="stretch")
        else:
            st.info("Need at least 2 numerical features for correlation analysis")

    with tab4:
        st.markdown("### Missing Values Analysis")

        with st.spinner("Analyzing missing values..."):
            # Use sampling for large datasets
            if is_large:
                sample_df = df.sample(min(10000, len(df)), random_state=42)
                missing_data = pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Missing Count": sample_df.isnull().sum()
                        * (len(df) / len(sample_df)),
                        "Missing %": (
                            sample_df.isnull().sum() / len(sample_df) * 100
                        ).round(2),
                    }
                ).sort_values("Missing Count", ascending=False)
                missing_data["Missing Count"] = missing_data["Missing Count"].astype(
                    int
                )
                st.caption("†Estimated from 10,000 row sample")
            else:
                missing_data = pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Missing Count": df.isnull().sum(),
                        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
                    }
                ).sort_values("Missing Count", ascending=False)

        # Only show columns with missing values
        missing_data = missing_data[missing_data["Missing Count"] > 0]

        if len(missing_data) > 0:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(missing_data, width="stretch")

            with col2:
                # Enhanced missing values chart
                fig = px.bar(
                    missing_data,
                    x="Column",
                    y="Missing %",
                    title="Missing Values by Column",
                    labels={"Missing %": "Missing Percentage (%)"},
                    color="Missing %",
                    color_continuous_scale=["#5ee7df", "#f093fb", "#f5576c"],
                )

                fig.update_traces(
                    marker=dict(line=dict(color="white", width=1.5)),
                    texttemplate="%{y:.1f}%",
                    textposition="outside",
                )

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    height=500,
                    showlegend=False,
                    xaxis=dict(
                        showgrid=False,
                        tickangle=-45,
                        title_font=dict(size=14, color="#666"),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(200,200,200,0.2)",
                        title_font=dict(size=14, color="#666"),
                    ),
                    title=dict(
                        font=dict(size=16, color="#333"), x=0.5, xanchor="center"
                    ),
                )

                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.plotly_chart(fig, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.success("✅ No missing values found in the dataset!")


def train_models_page():
    """Model training configuration and execution page."""

    # Professional Header
    st.markdown(
        """
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5rem;'>🎯 Train AutoML Models</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem;'>
            Configure and train your machine learning models
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if any data type is loaded
    has_tabular_data = st.session_state.data is not None
    has_image_data = st.session_state.get("image_data") is not None
    has_text_data = st.session_state.get("text_data") is not None

    if not (has_tabular_data or has_image_data or has_text_data):
        st.warning("⚠️ Please upload data first!")
        return

    # Set data type and df based on what's loaded
    # Prioritize image/text data over tabular (user's most recent upload intent)
    if has_image_data:
        data_type = "image"
        df = None
    elif has_text_data:
        data_type = "text"
        df = None
    else:
        data_type = "tabular"
        df = st.session_state.data

    # Only show text column warnings for tabular data
    if data_type == "tabular" and df is not None:
        # Check for text columns and warn user
        text_like_columns = []
        for col in df.columns:
            if df[col].dtype == "object":
                avg_length = df[col].astype(str).str.len().mean()
                unique_ratio = df[col].nunique() / len(df)
                if avg_length > 50 or unique_ratio > 0.5:
                    text_like_columns.append(col)

        if text_like_columns:
            error_msg = f"""
🚨 **Text Data Detected!**

Your dataset contains text columns with high cardinality:
**{', '.join(text_like_columns)}**

**⚠️ Do NOT use Traditional ML for text data!**

Please go back to **📁 Data Upload** and:
1. Select **📝 Text Data** option
2. Upload your CSV with text columns
3. Select which column contains the text
4. Select which column contains the labels
5. Use **🔶 Deep Learning (NLP models)** for training

Traditional ML will one-hot encode text, creating 10,000+ features and causing:
- ❌ Extreme memory usage
- ❌ Very slow training
- ❌ App freezing/crashing
- ❌ Poor prediction accuracy
"""
            st.error(error_msg)

            # Quick fix option: Drop text columns
            st.markdown("### 🛠️ Quick Fix Options")
            col_fix1, col_fix2 = st.columns(2)

            with col_fix1:
                if st.button("🗑️ Drop Text Columns & Continue", type="primary"):
                    # Drop text columns and continue
                    df = df.drop(columns=text_like_columns)
                    st.session_state.data = df
                    st.success(
                        f"✅ Dropped {len(text_like_columns)} text columns: {', '.join(text_like_columns)}"
                    )
                    st.info(f"Remaining columns: {', '.join(df.columns.tolist())}")
                    st.rerun()

            with col_fix2:
                if st.button("🔄 Switch to Text Data Upload"):
                    st.info(
                        "Please manually select 📝 Text Data from the sidebar navigation"
                    )

            st.markdown("---")

            if not st.checkbox(
                "⚠️ I understand the risks and want to proceed anyway", value=False
            ):
                st.stop()
            else:
                st.warning(
                    "Proceeding at your own risk. Training may fail or take very long."
                )

    # For image/text data, skip directly to deep learning configuration
    if data_type in ["image", "text"]:
        st.markdown(
            """
        <div style='margin: 2rem 0 1.5rem 0;'>
            <h2 style='font-size: 1.8rem; color: #2c3e50;'>⚙️ Deep Learning Training</h2>
            <p style='color: #7f8c8d; margin: 0.5rem 0;'>
                Configure and train your neural network
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Set variables for DL flow
        use_deep_learning = True
        target_column = None
        problem_type = "classification"
        model_family = "🔶 Deep Learning (PyTorch)"
        test_size = 0.2
        selection_mode = "🤖 Auto (Recommended)"  # Always use auto mode for image/text

        # Show brief info based on data type
        if data_type == "image":
            image_data = st.session_state.image_data
            st.success(
                f"📸 Ready to train on {len(image_data['files'])} images from {len(set(image_data['labels']))} classes"
            )
        elif data_type == "text":
            text_data = st.session_state.text_data
            if "dataframe" in text_data:
                st.success(
                    f"📝 Ready to train on {len(text_data['dataframe'])} text samples"
                )
            else:
                st.success(
                    f"📝 Ready to train on {len(text_data['texts'])} text samples"
                )

    # Tabular data configuration
    elif data_type == "tabular" and df is not None:
        # Configuration Section Header
        st.markdown(
            """
        <div style='margin: 2rem 0 1.5rem 0;'>
            <h2 style='font-size: 1.8rem; color: #2c3e50;'>⚙️ Training Configuration</h2>
            <p style='color: #7f8c8d; margin: 0.5rem 0;'>
                Set up your target variable and optimization parameters
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Training configuration
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Target Variable")
            target_column = st.selectbox(
                "Select target column",
                options=df.columns.tolist(),
                help="The column you want to predict",
            )

        # Analyze target column to suggest problem type
        if target_column:
            target_data = df[target_column]
            n_unique = target_data.nunique()
            n_samples = len(target_data)
            dtype = target_data.dtype

            # Smart problem type detection
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            is_categorical = dtype == "object" or dtype.name == "category"

            # Detection logic
            suggested_type = "auto"
            reasoning = []

            if is_categorical:
                suggested_type = "classification"
                reasoning.append(f"📊 Categorical dtype detected ({dtype})")
                reasoning.append(f"🎯 {n_unique} unique classes found")
            elif is_numeric:
                # Check uniqueness ratio
                unique_ratio = n_unique / n_samples

                if n_unique <= 20:
                    suggested_type = "classification"
                    reasoning.append(
                        f"🎯 Only {n_unique} unique values (likely classes)"
                    )
                    reasoning.append(f"📊 Numeric but discrete")
                elif unique_ratio < 0.05:
                    suggested_type = "classification"
                    reasoning.append(
                        f"🎯 {n_unique} unique values ({unique_ratio:.1%} of samples)"
                    )
                    reasoning.append(f"📊 Low cardinality suggests classification")
                else:
                    suggested_type = "regression"
                    reasoning.append(
                        f"📈 {n_unique} unique values ({unique_ratio:.1%} of samples)"
                    )
                    reasoning.append(f"📊 High cardinality suggests regression")

                    # Check if values are continuous
                    if target_data.dtype in ["float64", "float32"]:
                        reasoning.append("✓ Float dtype indicates continuous target")
            else:
                reasoning.append("❓ Unknown data type")

            # Display analysis
            with st.expander("🔍 Target Analysis", expanded=True):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Unique Values", n_unique)
                with col_b:
                    st.metric("Data Type", str(dtype))
                with col_c:
                    uniqueness_pct = (n_unique / n_samples) * 100
                    st.metric("Uniqueness", f"{uniqueness_pct:.1f}%")

                # Show reasoning
                st.markdown("**Detection Analysis:**")
                for reason in reasoning:
                    st.markdown(f"- {reason}")

                # Show sample values
                st.markdown("**Sample Values:**")
                sample_values = target_data.dropna().head(10).tolist()
                st.code(str(sample_values), language=None)

                # Classification-specific info
                if suggested_type == "classification" and n_unique <= 50:
                    st.markdown("**Class Distribution:**")
                    with st.spinner("Computing class distribution..."):
                        # For large datasets, use faster method
                        if len(target_data) > 100000:
                            class_counts = target_data.value_counts(
                                normalize=False
                            ).head(10)
                        else:
                            class_counts = target_data.value_counts().head(10)
                    st.bar_chart(class_counts)

            # Problem type selection with smart default
            problem_type_options = ["auto", "classification", "regression"]

            # Set default based on suggestion
            if suggested_type in problem_type_options:
                default_index = problem_type_options.index(suggested_type)
            else:
                default_index = 0

            problem_type = st.selectbox(
                "Problem Type",
                options=problem_type_options,
                index=default_index,
                help=f"💡 Suggested: {suggested_type.upper()} - {reasoning[0] if reasoning else 'Auto-detect'}",
            )

            # Show confirmation or warning
            if problem_type == suggested_type or problem_type == "auto":
                st.success(f"✅ Using **{problem_type.upper()}** (recommended)")
            else:
                st.warning(
                    f"⚠️ Manual override: {problem_type.upper()} (suggested: {suggested_type.upper()})"
                )
        else:
            # Fallback if no target selected
            problem_type = st.selectbox(
                "Problem Type",
                options=["auto", "classification", "regression"],
                help="Classification for categorical targets, regression for continuous",
            )

        use_cv = st.checkbox(
            "Use Cross-Validation",
            value=False,
            help="Use k-fold cross-validation for more robust evaluation",
        )

        if use_cv:
            cv_folds = st.slider("Number of folds", min_value=3, max_value=10, value=5)
        else:
            cv_folds = 5

        with col2:
            st.markdown("#### ⚡ Optimization Settings")

            optimize_hp = st.checkbox(
                "Enable Hyperparameter Optimization",
                value=True,
                help="Use Bayesian optimization to find best hyperparameters",
            )

        if optimize_hp:
            n_trials = st.slider(
                "Number of optimization trials",
                min_value=10,
                max_value=100,
                value=20,
                step=10,
                help="More trials = better results but longer training time",
            )
        else:
            n_trials = 20

        enable_mlflow = st.checkbox(
            "Enable MLflow Tracking",
            value=True,
            help="Track experiments and log models to MLflow",
        )

        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing",
        )

        st.markdown("---")

        # Model Selection Configuration
        st.markdown("### 🎯 Model Selection")
        st.markdown("Choose which models to train and compare.")

        # Get available models based on problem type
        # Use session state target and problem detection to get available models
        available_models = []
        if "target_column" in locals() and target_column:
            # Determine problem type for model listing
            if problem_type == "auto":
                # Simple auto-detection for model listing
                target_data = df[target_column]
                n_unique = target_data.nunique()
                is_numeric = pd.api.types.is_numeric_dtype(target_data)

                if not is_numeric or n_unique <= 20:
                    detected_for_models = "classification"
                else:
                    detected_for_models = "regression"
            else:
                detected_for_models = problem_type

            # Import model metadata to get available models
            try:
                from automl.models.model_metadata import list_models_by_type

                available_models = list_models_by_type(detected_for_models)
                st.caption(
                    f"📊 Available models for **{detected_for_models}**: {len(available_models)} models"
                )
            except Exception as e:
                st.warning(f"Could not load available models: {e}")
                available_models = []

        # Model selection mode
        model_selection_mode = st.radio(
            "Model Selection Mode:",
            ["🤖 Auto (Top 3 Recommended)", "🌟 All Models", "🎯 Custom Selection"],
            horizontal=True,
            help="Auto: Train top 3 recommended models | All: Train all available models | Custom: Choose specific models",
        )

        models_to_train = None  # Default: None means auto recommendation

        if model_selection_mode == "🌟 All Models":
            models_to_train = "all"  # Special marker for all models
            if available_models:
                st.info(
                    f"✅ Will train **all {len(available_models)} models** for comparison"
                )
            else:
                st.info("✅ Will train all available models for your problem type")

        elif model_selection_mode == "🎯 Custom Selection":
            if available_models:
                # Show multiselect with available models
                selected_models = st.multiselect(
                    "Select models to train:",
                    options=available_models,
                    default=(
                        available_models[:3]
                        if len(available_models) >= 3
                        else available_models
                    ),
                    help="Choose one or more models to train and compare",
                )

                if selected_models:
                    models_to_train = selected_models
                    st.success(
                        f"✅ Selected **{len(selected_models)} models** to train"
                    )
                else:
                    st.warning(
                        "⚠️ No models selected. Will use auto recommendation (top 3)."
                    )
                    models_to_train = None
            else:
                st.warning(
                    "⚠️ Could not load model list. Will use auto recommendation."
                )
                models_to_train = None

        else:  # Auto mode
            st.info(
                "🤖 Will automatically select **top 3 recommended models** based on dataset characteristics"
            )
            models_to_train = None  # None triggers auto recommendation

        # Store in session state
        st.session_state.models_to_train = models_to_train

    # For image/text data, set models_to_train to None (not used for DL)
    if data_type in ["image", "text"]:
        models_to_train = None
        st.session_state.models_to_train = None

    st.markdown("---")

    # Initialize preprocessing config
    preprocessing_config = {}

    # Advanced Preprocessing Configuration (only for tabular data)
    if data_type == "tabular" and df is not None:
        st.markdown("### 🔧 Advanced Preprocessing (Optional)")
        st.markdown(
            "Configure advanced data preprocessing features for better model performance."
        )

        # Advanced Imputation
        with st.expander("🔄 Advanced Missing Value Imputation"):
            use_advanced_imputation = st.checkbox(
                "Enable Advanced Imputation",
                value=False,
                help="Use KNN or Iterative (MICE) imputation instead of simple mean/median",
                key="use_adv_imp",
            )

            if use_advanced_imputation:
                imp_method = st.selectbox(
                    "Imputation Method",
                    options=["knn", "iterative", "mice"],
                    help="KNN uses k-nearest neighbors, Iterative/MICE uses multivariate imputation",
                )
                preprocessing_config["imputation_method"] = imp_method

                if imp_method == "knn":
                    n_neighbors = st.slider("Number of neighbors", 3, 15, 5)
                    preprocessing_config["n_neighbors"] = n_neighbors
                else:
                    max_iter = st.slider("Maximum iterations", 5, 20, 10)
                    preprocessing_config["max_iter"] = max_iter

        # Outlier Detection
        with st.expander("📊 Outlier Detection & Handling"):
            use_outlier_detection = st.checkbox(
                "Enable Outlier Detection",
                value=False,
                help="Detect and handle outliers using statistical or ML methods",
                key="use_outlier",
            )

            if use_outlier_detection:
                outlier_method = st.selectbox(
                    "Detection Method",
                    options=["iqr", "zscore", "isolation_forest"],
                    help="IQR: Interquartile Range, Z-Score: Standard Deviation, Isolation Forest: ML-based",
                )
                preprocessing_config["outlier_method"] = outlier_method

                outlier_action = st.selectbox(
                    "Action",
                    options=["remove", "cap", "flag"],
                    help="Remove rows, cap values (Winsorization), or add a flag column",
                )
                preprocessing_config["outlier_action"] = outlier_action

                if outlier_method == "iqr":
                    iqr_threshold = st.slider("IQR Threshold", 1.0, 3.0, 1.5, 0.1)
                    preprocessing_config["iqr_threshold"] = iqr_threshold
                elif outlier_method == "zscore":
                    z_threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
                    preprocessing_config["z_threshold"] = z_threshold
                else:
                    contamination = st.slider("Contamination", 0.01, 0.3, 0.1, 0.01)
                    preprocessing_config["contamination"] = contamination

        # Feature Engineering
        with st.expander("⚙️ Feature Engineering"):
            use_feature_engineering = st.checkbox(
                "Enable Feature Engineering",
                value=False,
                help="Create new features from existing ones",
                key="use_feat_eng",
            )

            if use_feature_engineering:
                st.markdown("**Select transformation types:**")

                create_polynomial = st.checkbox(
                    "Polynomial Features", help="Create x², x³, etc."
                )
                if create_polynomial:
                    poly_degree = st.slider("Polynomial Degree", 2, 3, 2)
                    preprocessing_config["poly_degree"] = poly_degree
                    preprocessing_config["poly_interaction_only"] = st.checkbox(
                        "Interaction terms only (no x², x³)", value=False
                    )

                create_interactions = st.checkbox(
                    "Feature Interactions", help="Create x*y interaction terms"
                )
                preprocessing_config["create_interactions"] = create_interactions

                create_transformations = st.checkbox(
                    "Mathematical Transformations", help="log, sqrt, square, inverse"
                )
                if create_transformations:
                    transform_types = st.multiselect(
                        "Transformation types",
                        options=["log", "sqrt", "square", "inverse"],
                        default=["log"],
                    )
                    preprocessing_config["transformations"] = transform_types

                create_binning = st.checkbox(
                    "Binning/Discretization", help="Convert continuous to categorical"
                )
                if create_binning:
                    n_bins = st.slider("Number of bins", 3, 10, 5)
                    bin_strategy = st.selectbox(
                        "Binning strategy", ["quantile", "uniform", "kmeans"]
                    )
                    preprocessing_config["n_bins"] = n_bins
                    preprocessing_config["bin_strategy"] = bin_strategy

        # Feature Selection
        with st.expander("🎯 Feature Selection"):
            use_feature_selection = st.checkbox(
                "Enable Feature Selection",
                value=False,
                help="Select most relevant features to improve model performance",
                key="use_feat_sel",
            )

            if use_feature_selection:
                selection_method = st.selectbox(
                    "Selection Method",
                    options=[
                        "correlation",
                        "mutual_info",
                        "anova_f",
                        "chi2",
                        "rfe",
                        "l1",
                        "tree_importance",
                    ],
                    help="Different methods for ranking feature importance",
                )
                preprocessing_config["selection_method"] = selection_method

                k_features = st.slider(
                    "Number of features to select",
                    min_value=5,
                    max_value=min(
                        50, len([c for c in df.columns if c != target_column])
                    ),
                    value=10,
                    help="How many top features to keep",
                )
                preprocessing_config["k_features"] = k_features

                if selection_method == "correlation":
                    corr_threshold = st.slider(
                        "Correlation Threshold", 0.0, 1.0, 0.5, 0.05
                    )
                    preprocessing_config["correlation_threshold"] = corr_threshold

        # Store preprocessing config in session state
        st.session_state.preprocessing_config = preprocessing_config

        st.markdown("---")

        # Show preprocessing summary if any features are enabled
        if preprocessing_config:
            st.markdown("### 📋 Preprocessing Summary")
            enabled_features = []

            if "imputation_method" in preprocessing_config:
                enabled_features.append(
                    f"✓ Advanced Imputation ({preprocessing_config['imputation_method'].upper()})"
                )
            if "outlier_method" in preprocessing_config:
                enabled_features.append(
                    f"✓ Outlier Detection ({preprocessing_config['outlier_method'].upper()}, action: {preprocessing_config.get('outlier_action', 'remove')})"
                )
            if "poly_degree" in preprocessing_config:
                enabled_features.append(
                    f"✓ Polynomial Features (degree {preprocessing_config['poly_degree']})"
                )
            if preprocessing_config.get("create_interactions"):
                enabled_features.append("✓ Feature Interactions")
            if "transformations" in preprocessing_config:
                enabled_features.append(
                    f"✓ Transformations ({', '.join(preprocessing_config['transformations'])})"
                )
            if "n_bins" in preprocessing_config:
                enabled_features.append(
                    f"✓ Binning ({preprocessing_config.get('bin_strategy', 'quantile')}, {preprocessing_config['n_bins']} bins)"
                )
            if "selection_method" in preprocessing_config:
                enabled_features.append(
                    f"✓ Feature Selection ({preprocessing_config['selection_method']}, top {preprocessing_config.get('k_features', 10)})"
                )

            if enabled_features:
                st.info(
                    "**Enabled Preprocessing Features:**\n\n"
                    + "\n\n".join(enabled_features)
                )

            st.markdown("---")

        # ML vs DL Recommendation (for tabular data)
        st.markdown("### 🤖 Training Approach Recommendation")

        # Safely get target_column and problem_type (they should be defined by now)
        current_target = locals().get("target_column", None)
        current_problem_type = locals().get("problem_type", None)

        # Get recommendation
        recommendation_result = get_ml_vs_dl_recommendation(
            df=df,
            target_column=current_target,
            problem_type=current_problem_type,
            data_type="tabular",
        )

        # Display recommendation box with visual styling
        rec = recommendation_result["recommendation"]
        confidence = recommendation_result["confidence"]
        score = recommendation_result["score"]
        reasons = recommendation_result["reasons"]
        details = recommendation_result["details"]

        # Color coding based on recommendation
        if rec == "dl":
            box_color = "#FF9F66"  # Orange for DL
            rec_emoji = "🔶"
            rec_text = "Deep Learning (PyTorch)"
            box_style = "background: linear-gradient(135deg, #ff9f66 0%, #ff6b6b 100%);"
        else:
            box_color = "#66B2FF"  # Blue for ML
            rec_emoji = "🔷"
            rec_text = "Traditional ML (AutoML)"
            box_style = "background: linear-gradient(135deg, #66b2ff 0%, #667eea 100%);"

        # Confidence indicator
        if confidence == "high":
            confidence_emoji = "✅"
            confidence_text = "High Confidence"
        elif confidence == "medium":
            confidence_emoji = "⚖️"
            confidence_text = "Medium Confidence"
        else:
            confidence_emoji = "❓"
            confidence_text = "Low Confidence - Either approach viable"

        # Create recommendation box
        st.markdown(
            f"""
        <div style="{box_style} padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h3 style="margin-top: 0; color: white;">💡 Recommended Approach</h3>
            <div style="font-size: 1.8rem; font-weight: bold; margin: 1rem 0;">
                {rec_emoji} {rec_text}
            </div>
            <div style="font-size: 1.1rem; margin-top: 0.5rem;">
                {confidence_emoji} {confidence_text} (Score: {score:.0f}/100)
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show detailed reasoning in an expander
        with st.expander("📊 View Analysis Details", expanded=False):
            st.markdown("####Why this recommendation?")
            for reason in reasons:
                st.markdown(f"- {reason}")

            st.markdown("#### Dataset Characteristics")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Samples", f"{details.get('n_samples', 0):,}")
            with col_b:
                st.metric("Features", f"{details.get('n_features', 0):,}")
            with col_c:
                if "n_classes" in details:
                    st.metric("Classes", details["n_classes"])
                else:
                    st.metric("Data Type", "Tabular")

            # Visual score bar
            st.markdown("#### Recommendation Score")
            st.caption("0 = Traditional ML ideal, 100 = Deep Learning ideal")

            # Create a visual score indicator
            score_normalized = score / 100
            if score < 40:
                bar_color = "#66B2FF"  # Blue for ML
                zone_text = "Traditional ML Zone"
            elif score > 60:
                bar_color = "#FF9F66"  # Orange for DL
                zone_text = "Deep Learning Zone"
            else:
                bar_color = "#FFD166"  # Yellow for neutral
                zone_text = "Either Approach Viable"

            st.progress(score_normalized)
            st.markdown(f"**{zone_text}** ({score:.0f}/100)")

        st.markdown("---")

        # Automatic Model Selection
        st.markdown("### 🧠 Model Selection")

        # Add auto mode option
        selection_mode = st.radio(
            "Selection Mode:",
            ["🤖 Auto (Recommended)", "⚙️ Manual Selection"],
            horizontal=True,
            help="Auto mode uses the recommendation above, Manual lets you override",
        )

        # Determine model family
        if selection_mode == "🤖 Auto (Recommended)":
            # Use recommendation automatically
            if rec == "dl":
                model_family = "🔶 Deep Learning (PyTorch)"
                st.info(
                    f"✅ **Automatic Selection**: Using {rec_text} (Confidence: {confidence_text})"
                )
            else:
                model_family = "🔷 Traditional ML (AutoML)"
                st.info(
                    f"✅ **Automatic Selection**: Using {rec_text} (Confidence: {confidence_text})"
                )
        else:
            # Manual override
            st.markdown(
                "Choose between traditional ML models and deep learning models."
            )
            model_family = st.radio(
                "Model Family:",
                ["🔷 Traditional ML (AutoML)", "🔶 Deep Learning (PyTorch)"],
                horizontal=True,
                help="Traditional ML uses scikit-learn models, Deep Learning uses PyTorch",
            )

            # Show warning if overriding recommendation
            if (rec == "dl" and "Traditional ML" in model_family) or (
                rec == "ml" and "Deep Learning" in model_family
            ):
                st.warning(
                    f"⚠️ Manual override: You're choosing a different approach than recommended ({rec_text})"
                )

        use_deep_learning = "Deep Learning" in model_family

    # For image/text data, use_deep_learning is already set to True (line 2071)
    # For tabular data, it's set above based on model_family selection

    # Common configuration for all data types (Hardware Status and DL Config)

    # Initialize DL config dictionary for all data types
    dl_config = {}

    # Show GPU status for all models
    st.markdown("### 💻 Hardware Status")

    try:
        import torch

        gpu_available = torch.cuda.is_available()

        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GPU Status", "✅ Available", delta="Enabled")
            with col2:
                st.metric("GPU Name", gpu_name)
            with col3:
                st.metric("CUDA Version", cuda_version)

            # Show which models support GPU
            if use_deep_learning:
                st.success("🚀 Deep Learning models will use GPU acceleration!")
            else:
                st.info(
                    "⚡ XGBoost, LightGBM, and CatBoost will use GPU acceleration automatically!"
                )
        else:
            st.warning("⚠️ GPU not detected - All models will use CPU")
            st.caption("💡 To enable GPU: Install CUDA toolkit and GPU-enabled PyTorch")
    except ImportError:
        st.warning("⚠️ PyTorch not detected - Cannot check GPU status")

    st.markdown("---")

    if use_deep_learning:
        # Check data type and PyTorch availability
        data_type = st.session_state.get("data_type", "tabular")

        try:
            import torch

            torch_available = True
            gpu_available = torch.cuda.is_available()

            # Get detailed GPU info
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
            else:
                gpu_count = 0
                gpu_name = "N/A"
                cuda_version = "N/A"
        except:
            torch_available = False
            gpu_available = False
            gpu_count = 0
            gpu_name = "N/A"
            cuda_version = "N/A"

        if not torch_available:
            st.error(
                "❌ PyTorch not installed! Please install: `pip install torch torchvision`"
            )
            return

        # Display GPU status with details
        if gpu_available:
            st.success(
                f"✅ PyTorch available | GPU: **{gpu_name}** (CUDA {cuda_version}) | Count: {gpu_count}"
            )
            st.info(
                "💡 Deep Learning models will automatically use GPU acceleration for faster training!"
            )
        else:
            st.warning(
                "⚠️ GPU not detected - Deep Learning models will use CPU (slower training)"
            )

        # Auto-detect model type based on data type (for Auto mode)
        if selection_mode == "🤖 Auto (Recommended)":
            st.markdown(
                f"#### 🎯 Auto-Selecting Best DL Architecture", unsafe_allow_html=True
            )

            if data_type == "image":
                st.info(
                    "🖼️ **Analyzing image dataset to select optimal CNN architecture...**"
                )

                # Get image data characteristics
                image_data = st.session_state.get("image_data")
                if image_data:
                    num_images = len(image_data["files"])
                    num_classes = len(set(image_data["labels"]))

                    # Smart architecture selection based on dataset size and complexity
                    if num_images < 500:
                        # Small dataset - use simple CNN to avoid overfitting
                        recommended_arch = "simple"
                        reason = f"Small dataset ({num_images} images) - Simple CNN prevents overfitting"
                        dl_config["max_epochs"] = 30
                    elif num_images < 2000:
                        # Medium dataset - use ResNet18
                        recommended_arch = "resnet18"
                        reason = f"Medium dataset ({num_images} images) - ResNet18 balances performance and efficiency"
                        dl_config["max_epochs"] = 40
                    elif num_images < 10000:
                        # Large dataset - use ResNet50
                        recommended_arch = "resnet50"
                        reason = f"Large dataset ({num_images} images) - ResNet50 for better accuracy"
                        dl_config["max_epochs"] = 50
                    else:
                        # Very large dataset - use EfficientNet
                        recommended_arch = "efficientnet_b0"
                        reason = f"Very large dataset ({num_images} images) - EfficientNet for optimal efficiency"
                        dl_config["max_epochs"] = 50

                    # Adjust batch size based on architecture
                    if recommended_arch in ["efficientnet_b0", "resnet50"]:
                        dl_config["batch_size"] = 16 if gpu_available else 8
                    else:
                        dl_config["batch_size"] = 32 if gpu_available else 16

                    dl_config["architecture"] = recommended_arch
                    dl_config["image_size"] = 224
                    dl_config["learning_rate"] = 0.001

                    st.success(f"✅ **Selected: {recommended_arch.upper()}**")
                    st.caption(f"📊 {reason}")
                    st.caption(
                        f"🏷️ Classes: {num_classes} | 🖼️ Images: {num_images:,} | 📦 Batch Size: {dl_config['batch_size']} | 🔄 Epochs: {dl_config['max_epochs']}"
                    )
                else:
                    # Fallback
                    dl_config["architecture"] = "resnet18"
                    dl_config["image_size"] = 224
                    dl_config["batch_size"] = 32
                    dl_config["max_epochs"] = 50
                    dl_config["learning_rate"] = 0.001

            elif data_type == "text":
                st.info(
                    "📝 **Analyzing text dataset to select optimal NLP architecture...**"
                )

                # Get text data characteristics
                text_data = st.session_state.get("text_data")
                if text_data:
                    # Determine dataset size
                    if text_data.get("is_large", False):
                        num_texts = len(text_data["dataframe"])
                        texts_sample = (
                            text_data["dataframe"][text_data["text_column"]]
                            .head(1000)
                            .tolist()
                        )
                    else:
                        num_texts = len(text_data["texts"])
                        texts_sample = text_data["texts"][:1000]

                    # Calculate average text length
                    avg_length = sum(len(str(t).split()) for t in texts_sample) / len(
                        texts_sample
                    )

                    # Smart architecture selection
                    if num_texts < 1000:
                        # Small dataset - use LSTM
                        recommended_arch = "lstm"
                        reason = f"Small dataset ({num_texts:,} texts) - LSTM is robust"
                        dl_config["max_epochs"] = 30
                    elif avg_length < 50:
                        # Short texts - use LSTM or GRU
                        recommended_arch = "gru"
                        reason = f"Short texts (avg {avg_length:.0f} words) - GRU is efficient"
                        dl_config["max_epochs"] = 40
                    elif num_texts > 10000 and avg_length > 100:
                        # Large dataset with long texts - use Transformer
                        recommended_arch = "transformer"
                        reason = f"Large dataset ({num_texts:,} texts, avg {avg_length:.0f} words) - Transformer for best performance"
                        dl_config["max_epochs"] = 30
                    else:
                        # Default to LSTM
                        recommended_arch = "lstm"
                        reason = f"Balanced dataset ({num_texts:,} texts, avg {avg_length:.0f} words) - LSTM is reliable"
                        dl_config["max_epochs"] = 40

                    # Adjust batch size
                    if recommended_arch == "transformer":
                        dl_config["batch_size"] = 16 if gpu_available else 8
                    else:
                        dl_config["batch_size"] = 32 if gpu_available else 16

                    dl_config["architecture"] = recommended_arch
                    dl_config["embedding_dim"] = 128
                    dl_config["hidden_dim"] = 128
                    dl_config["learning_rate"] = 0.001

                    st.success(f"✅ **Selected: {recommended_arch.upper()}**")
                    st.caption(f"📊 {reason}")
                    st.caption(
                        f"📝 Texts: {num_texts:,} | 📏 Avg Length: {avg_length:.0f} words | 📦 Batch Size: {dl_config['batch_size']} | 🔄 Epochs: {dl_config['max_epochs']}"
                    )
                else:
                    # Fallback
                    dl_config["architecture"] = "lstm"
                    dl_config["embedding_dim"] = 128
                    dl_config["hidden_dim"] = 128
                    dl_config["batch_size"] = 32
                    dl_config["max_epochs"] = 30
                    dl_config["learning_rate"] = 0.001

            else:  # tabular
                st.info(
                    "📊 **Analyzing tabular dataset to select optimal architecture...**"
                )

                # Get tabular data characteristics
                if df is not None:
                    num_samples = len(df)
                    num_features = len(df.columns) - 1  # Exclude target

                    # Smart architecture selection
                    if num_features < 10:
                        # Few features - use simple MLP
                        hidden_dims = [64, 32]
                        reason = f"Few features ({num_features}) - Simple 2-layer MLP"
                    elif num_features < 50:
                        # Medium features - use standard MLP
                        hidden_dims = [128, 64, 32]
                        reason = f"Medium features ({num_features}) - 3-layer MLP"
                    elif num_features < 200:
                        # Many features - use deeper MLP
                        hidden_dims = [256, 128, 64, 32]
                        reason = f"Many features ({num_features}) - 4-layer deep MLP"
                    else:
                        # Very many features - use very deep MLP
                        hidden_dims = [512, 256, 128, 64, 32]
                        reason = (
                            f"Very many features ({num_features}) - 5-layer deep MLP"
                        )

                    # Adjust epochs based on dataset size
                    if num_samples < 1000:
                        max_epochs = 30
                    elif num_samples < 10000:
                        max_epochs = 50
                    else:
                        max_epochs = 30  # Large datasets learn faster

                    dl_config["architecture"] = "mlp"
                    dl_config["hidden_dims"] = hidden_dims
                    dl_config["activation"] = "relu"
                    dl_config["dropout"] = 0.3
                    dl_config["batch_size"] = 32
                    dl_config["max_epochs"] = max_epochs
                    dl_config["learning_rate"] = 0.001

                    st.success(f"✅ **Selected: MLP with {len(hidden_dims)} layers**")
                    st.caption(f"📊 {reason}")
                    st.caption(
                        f"🔢 Layers: {' → '.join(map(str, hidden_dims))} | 📦 Samples: {num_samples:,} | 🔄 Epochs: {max_epochs}"
                    )
                else:
                    # Fallback
                    dl_config["architecture"] = "mlp"
                    dl_config["hidden_dims"] = [128, 64, 32]
                    dl_config["activation"] = "relu"
                    dl_config["dropout"] = 0.3
                    dl_config["batch_size"] = 32
                    dl_config["max_epochs"] = 50
                    dl_config["learning_rate"] = 0.001

            # Show advanced settings in expander
            with st.expander("⚙️ Advanced Deep Learning Settings (Optional)"):
                # Override checkbox to enable customization
                override_auto = st.checkbox(
                    "🔧 Override Auto Settings",
                    value=False,
                    help="Enable to customize the auto-selected configuration. Leave unchecked to use optimized defaults.",
                    key="override_auto_settings",
                )

                if not override_auto:
                    st.info(
                        "ℹ️ Auto-selected settings are locked for optimal performance. Check the box above to customize."
                    )

                st.markdown(
                    "**Adjust auto-selected configuration or enable multi-architecture training:**"
                )

                # Option to train multiple architectures
                st.markdown("---")
                train_multiple = st.checkbox(
                    "🔬 Train Multiple Architectures & Auto-Select Best",
                    value=False,
                    help="Train 2-3 architectures with reduced epochs and automatically select the best performer (like traditional AutoML)",
                    disabled=not override_auto,
                )

                if train_multiple:
                    st.session_state.train_multiple_dl = True
                    st.info(
                        "✅ Will train multiple DL architectures and select the best based on validation performance"
                    )

                    # Determine which architectures to try
                    if data_type == "image":
                        all_image_archs = [
                            "simple",
                            "medium",
                            "resnet18",
                            "resnet50",
                            "vgg16",
                            "vgg19",
                            "densenet121",
                            "densenet169",
                            "efficientnet_b0",
                        ]
                        selected_archs = st.multiselect(
                            "Select Architectures to Train",
                            options=all_image_archs,
                            default=["simple", "resnet18", "resnet50"],
                            help="Choose 2+ architectures for comparison. Default: Simple, ResNet18, ResNet50",
                        )

                        if len(selected_archs) >= 2:
                            dl_config["architectures_to_try"] = selected_archs
                            arch_display = ", ".join(
                                [a.upper().replace("_", " ") for a in selected_archs]
                            )
                            st.caption(f"📋 Will try: {arch_display}")
                        else:
                            st.warning(
                                "⚠️ Please select at least 2 architectures for comparison"
                            )
                            dl_config["architectures_to_try"] = [
                                "simple",
                                "resnet18",
                                "resnet50",
                            ]

                    elif data_type == "text":
                        all_text_archs = ["lstm", "gru", "transformer"]
                        selected_archs = st.multiselect(
                            "Select Architectures to Train",
                            options=all_text_archs,
                            default=["lstm", "gru", "transformer"],
                            help="Choose 2+ architectures for comparison",
                        )

                        if len(selected_archs) >= 2:
                            dl_config["architectures_to_try"] = selected_archs
                            arch_display = ", ".join(
                                [a.upper() for a in selected_archs]
                            )
                            st.caption(f"📋 Will try: {arch_display}")
                        else:
                            st.warning(
                                "⚠️ Please select at least 2 architectures for comparison"
                            )
                            dl_config["architectures_to_try"] = [
                                "lstm",
                                "gru",
                                "transformer",
                            ]

                    else:  # tabular
                        all_tabular_archs = ["mlp_small", "mlp_medium", "mlp_deep"]
                        selected_archs = st.multiselect(
                            "Select MLP Configurations to Train",
                            options=all_tabular_archs,
                            default=["mlp_small", "mlp_medium", "mlp_deep"],
                            help="Choose 2+ MLP configurations for comparison",
                        )

                        if len(selected_archs) >= 2:
                            dl_config["architectures_to_try"] = selected_archs
                            arch_names = {
                                "mlp_small": "Small MLP (2 layers)",
                                "mlp_medium": "Medium MLP (3 layers)",
                                "mlp_deep": "Deep MLP (4 layers)",
                            }
                            arch_display = ", ".join(
                                [arch_names[a] for a in selected_archs]
                            )
                            st.caption(f"📋 Will try: {arch_display}")
                        else:
                            st.warning(
                                "⚠️ Please select at least 2 configurations for comparison"
                            )
                            dl_config["architectures_to_try"] = [
                                "mlp_small",
                                "mlp_medium",
                                "mlp_deep",
                            ]

                    # Reduce epochs for quick comparison
                    quick_epochs = st.slider(
                        "Quick Training Epochs (per model)",
                        min_value=10,
                        max_value=30,
                        value=15,
                        help="Fewer epochs for quick comparison. Best model will be identified.",
                        disabled=not override_auto,
                    )
                    dl_config["quick_epochs"] = quick_epochs
                else:
                    st.session_state.train_multiple_dl = False

                st.markdown("---")
                st.markdown("**Individual Settings:**")

                if data_type == "tabular":
                    col1, col2 = st.columns(2)
                    with col1:
                        # Architecture options
                        arch_options = ["mlp", "resnet", "attention", "wide_deep"]
                        current_arch = dl_config.get("architecture", "mlp")
                        try:
                            default_idx = arch_options.index(current_arch)
                        except ValueError:
                            default_idx = 0

                        dl_config["architecture"] = st.selectbox(
                            "Model Architecture",
                            options=arch_options,
                            index=default_idx,
                            help="MLP: Standard, ResNet: Deep with skip connections, Attention: Transformer-based, Wide&Deep: Hybrid",
                            disabled=not override_auto,
                        )

                        hidden_dims_str = st.text_input(
                            "Hidden Dimensions",
                            value=",".join(map(str, dl_config["hidden_dims"])),
                            help="Comma-separated layer sizes",
                            disabled=not override_auto,
                        )
                        dl_config["hidden_dims"] = [
                            int(x.strip()) for x in hidden_dims_str.split(",")
                        ]

                        dl_config["activation"] = st.selectbox(
                            "Activation Function",
                            options=["relu", "leaky_relu", "elu", "tanh"],
                            index=0,
                            disabled=not override_auto,
                        )
                    with col2:
                        dl_config["dropout"] = st.slider(
                            "Dropout Rate",
                            0.0,
                            0.7,
                            dl_config["dropout"],
                            0.1,
                            disabled=not override_auto,
                        )
                        batch_size_override = st.slider(
                            "Batch Size",
                            16,
                            256,
                            dl_config.get("batch_size", 32),
                            16,
                            disabled=not override_auto,
                            key="dl_tabular_batch_size_override",
                        )
                        if override_auto:
                            dl_config["batch_size"] = batch_size_override
                        max_epochs_override = st.slider(
                            "Max Epochs",
                            10,
                            200,
                            dl_config.get("max_epochs", 50),
                            10,
                            disabled=not override_auto,
                            key="dl_tabular_max_epochs_override",
                        )
                        if override_auto:
                            dl_config["max_epochs"] = max_epochs_override

                elif data_type == "image":
                    col1, col2 = st.columns(2)
                    with col1:
                        # Architecture options
                        arch_options = [
                            "simple",
                            "medium",
                            "resnet18",
                            "resnet50",
                            "vgg16",
                            "vgg19",
                            "densenet121",
                            "densenet169",
                            "efficientnet_b0",
                        ]
                        # Find current architecture's index
                        current_arch = dl_config.get("architecture", "resnet18")
                        try:
                            default_idx = arch_options.index(current_arch)
                        except ValueError:
                            default_idx = 2  # fallback to resnet18

                        dl_config["architecture"] = st.selectbox(
                            "CNN Architecture",
                            options=arch_options,
                            index=default_idx,
                            help="Simple/Medium: Custom CNNs, ResNet/VGG/DenseNet/EfficientNet: Transfer Learning",
                            disabled=not override_auto,
                        )
                        dl_config["image_size"] = st.selectbox(
                            "Image Size",
                            options=[64, 128, 224, 256],
                            index=2,
                            disabled=not override_auto,
                        )
                    with col2:
                        batch_size_override = st.slider(
                            "Batch Size",
                            8,
                            128,
                            dl_config.get("batch_size", 32),
                            8,
                            disabled=not override_auto,
                            key="dl_image_batch_size_override",
                        )
                        if override_auto:
                            dl_config["batch_size"] = batch_size_override
                        max_epochs_override = st.slider(
                            "Max Epochs",
                            10,
                            100,
                            dl_config.get("max_epochs", 50),
                            10,
                            disabled=not override_auto,
                            key="dl_image_max_epochs_override",
                        )
                        if override_auto:
                            dl_config["max_epochs"] = max_epochs_override

                elif data_type == "text":
                    col1, col2 = st.columns(2)
                    with col1:
                        # Architecture options
                        arch_options = ["lstm", "gru", "transformer"]
                        current_arch = dl_config.get("architecture", "lstm")
                        try:
                            default_idx = arch_options.index(current_arch)
                        except ValueError:
                            default_idx = 0

                        dl_config["architecture"] = st.selectbox(
                            "Model Architecture",
                            options=arch_options,
                            index=default_idx,
                            disabled=not override_auto,
                        )
                        dl_config["embedding_dim"] = st.slider(
                            "Embedding Dim",
                            64,
                            512,
                            dl_config["embedding_dim"],
                            64,
                            disabled=not override_auto,
                        )
                    with col2:
                        batch_size_override = st.slider(
                            "Batch Size",
                            16,
                            128,
                            dl_config.get("batch_size", 32),
                            16,
                            disabled=not override_auto,
                            key="dl_text_batch_size_override",
                        )
                        if override_auto:
                            dl_config["batch_size"] = batch_size_override
                        max_epochs_override = st.slider(
                            "Max Epochs",
                            10,
                            100,
                            dl_config.get("max_epochs", 40),
                            10,
                            disabled=not override_auto,
                            key="dl_text_max_epochs_override",
                        )
                        if override_auto:
                            dl_config["max_epochs"] = max_epochs_override

        else:
            # Manual configuration (existing detailed UI)
            st.markdown("#### 🔧 Configure Deep Learning Model")

        # Model architecture selection based on data type (for Manual mode)
        if selection_mode == "⚙️ Manual Selection":
            if data_type == "tabular":
                st.markdown("#### 📊 MLP Configuration (Tabular Data)")

                col1, col2 = st.columns(2)

                with col1:
                    hidden_dims = st.text_input(
                        "Hidden Dimensions",
                        value="128,64,32",
                        help="Comma-separated layer sizes, e.g., 128,64,32",
                    )
                    dl_config["hidden_dims"] = [
                        int(x.strip()) for x in hidden_dims.split(",")
                    ]

                    activation = st.selectbox(
                        "Activation Function",
                        options=["relu", "leaky_relu", "elu", "tanh"],
                        help="Non-linear activation between layers",
                    )
                    dl_config["activation"] = activation

                    dropout = st.slider("Dropout Rate", 0.0, 0.7, 0.3, 0.1)
                    dl_config["dropout"] = dropout

                with col2:
                    batch_size = st.slider("Batch Size", 16, 256, 32, 16)
                    dl_config["batch_size"] = batch_size

                    max_epochs = st.slider("Max Epochs", 10, 200, 50, 10)
                    dl_config["max_epochs"] = max_epochs

                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001,
                    )
                    dl_config["learning_rate"] = learning_rate

            elif data_type == "image":
                st.markdown("#### 🖼️ CNN Configuration (Computer Vision)")

                col1, col2 = st.columns(2)

                with col1:
                    architecture = st.selectbox(
                        "CNN Architecture",
                        options=[
                            "simple",
                            "medium",
                            "resnet18",
                            "resnet50",
                            "vgg16",
                            "vgg19",
                            "densenet121",
                            "densenet169",
                            "efficientnet_b0",
                        ],
                        help="Simple/Medium: Custom CNNs, ResNet/VGG/DenseNet/EfficientNet: Transfer Learning",
                    )
                    dl_config["architecture"] = architecture

                    if "resnet" in architecture or "efficientnet" in architecture:
                        freeze_layers = st.slider(
                            "Freeze Layers",
                            0,
                            10,
                            5,
                            help="Number of initial layers to freeze during fine-tuning",
                        )
                        dl_config["freeze_layers"] = freeze_layers

                    image_size = st.selectbox(
                        "Image Size",
                        options=[64, 128, 224, 256],
                        index=2,
                        help="Input image resolution (224 recommended for transfer learning)",
                    )
                    dl_config["image_size"] = image_size

                with col2:
                    batch_size = st.slider("Batch Size", 8, 64, 16, 8)
                    dl_config["batch_size"] = batch_size

                    max_epochs = st.slider("Max Epochs", 10, 100, 30, 10)
                    dl_config["max_epochs"] = max_epochs

                    use_augmentation = st.checkbox(
                        "Data Augmentation",
                        value=True,
                        help="Random flips, rotations, color jitter",
                    )
                    dl_config["use_augmentation"] = use_augmentation

                    use_gpu = st.checkbox("Use GPU", value=gpu_available)
                    dl_config["use_gpu"] = use_gpu

            elif data_type == "text":
                st.markdown("#### 📝 NLP Configuration (Text Classification)")

                col1, col2 = st.columns(2)

                with col1:
                    architecture = st.selectbox(
                        "NLP Architecture",
                        options=["lstm", "gru", "cnn", "attention"],
                        help="LSTM/GRU: Sequential, CNN: Fast, Attention: Best accuracy",
                    )
                    dl_config["architecture"] = architecture

                    embedding_dim = st.slider("Embedding Dimension", 50, 300, 100, 50)
                    dl_config["embedding_dim"] = embedding_dim

                    hidden_dim = st.slider("Hidden Dimension", 64, 512, 128, 64)
                    dl_config["hidden_dim"] = hidden_dim

                    max_length = st.slider("Max Sequence Length", 50, 512, 128, 50)
                    dl_config["max_length"] = max_length

                with col2:
                    num_layers = st.slider("Number of Layers", 1, 4, 1)
                    dl_config["num_layers"] = num_layers

                    bidirectional = st.checkbox("Bidirectional", value=True)
                    dl_config["bidirectional"] = bidirectional

                    batch_size = st.slider("Batch Size", 8, 128, 32, 8)
                    dl_config["batch_size"] = batch_size

                    max_epochs = st.slider("Max Epochs", 5, 50, 20, 5)
                    dl_config["max_epochs"] = max_epochs

                    use_gpu = st.checkbox("Use GPU", value=gpu_available)
                    dl_config["use_gpu"] = use_gpu

        # Advanced DL settings (shown for both Auto and Manual modes)
        if selection_mode == "⚙️ Manual Selection":
            with st.expander("⚙️ Advanced Deep Learning Settings"):
                col1, col2 = st.columns(2)

                with col1:
                    optimizer = st.selectbox(
                        "Optimizer",
                        options=["adam", "adamw", "sgd", "rmsprop"],
                        help="AdamW recommended for best results",
                    )
                    dl_config["optimizer_name"] = optimizer

                    weight_decay = st.slider(
                        "Weight Decay (L2)",
                        0.0,
                        0.1,
                        0.01,
                        0.001,
                        help="L2 regularization strength",
                    )
                    dl_config["weight_decay"] = weight_decay

                    gradient_clip = st.slider(
                        "Gradient Clipping",
                        0.0,
                        5.0,
                        1.0,
                        0.5,
                        help="Max gradient norm (0 = disabled)",
                    )
                    if gradient_clip > 0:
                        dl_config["gradient_clip_value"] = gradient_clip

                with col2:
                    lr_scheduler = st.selectbox(
                        "LR Scheduler",
                        options=["plateau", "step", "cosine", "none"],
                        help="Learning rate scheduling strategy",
                    )
                    dl_config["lr_scheduler"] = lr_scheduler

                    early_stopping = st.slider(
                        "Early Stopping Patience",
                        0,
                        20,
                        5,
                        help="Stop training if no improvement for N epochs",
                    )
                    dl_config["early_stopping_patience"] = early_stopping

                    dl_config["random_state"] = 42

        # Set default advanced settings for Auto mode if not already set
        if selection_mode == "🤖 Auto (Recommended)":
            if "optimizer_name" not in dl_config:
                dl_config["optimizer_name"] = "adam"
            if "lr_scheduler" not in dl_config:
                dl_config["lr_scheduler"] = "plateau"
            if "early_stopping_patience" not in dl_config:
                dl_config["early_stopping_patience"] = 5
            if "weight_decay" not in dl_config:
                dl_config["weight_decay"] = 0.01
            dl_config["random_state"] = 42

        # Store DL config in session state
        st.session_state.dl_config = dl_config

        # Show DL summary
        if selection_mode == "🤖 Auto (Recommended)":
            st.markdown("#### 📋 Auto-Selected Configuration")
            st.caption(
                "✨ Optimized defaults will be used. Expand 'Advanced Settings' above to customize."
            )
        else:
            st.markdown("#### 📋 Deep Learning Configuration Summary")

        config_summary = []

        if data_type == "tabular":
            arch_name = dl_config.get("architecture", "mlp").upper()
            if arch_name == "MLP":
                hidden_layers_str = " → ".join(
                    map(str, dl_config.get("hidden_dims", [128, 64, 32]))
                )
                config_summary.append(
                    f"🔹 Architecture: {arch_name} ({hidden_layers_str})"
                )
            elif arch_name == "RESNET":
                config_summary.append(
                    f"🔹 Architecture: TabularResNet (with skip connections)"
                )
            elif arch_name == "ATTENTION":
                config_summary.append(f"🔹 Architecture: Transformer Attention")
            elif arch_name == "WIDE_DEEP":
                config_summary.append(f"🔹 Architecture: Wide & Deep")
            else:
                config_summary.append(f"🔹 Architecture: {arch_name}")
            config_summary.append(
                f"🔹 Activation: {dl_config.get('activation', 'relu').upper()}"
            )
            config_summary.append(f"🔹 Dropout: {dl_config.get('dropout', 0.3)}")
        elif data_type == "image":
            config_summary.append(
                f"🔹 Architecture: {dl_config.get('architecture', 'resnet18').upper()}"
            )
            config_summary.append(
                f"🔹 Image Size: {dl_config.get('image_size', 224)}×{dl_config.get('image_size', 224)}"
            )
            if dl_config.get("use_augmentation"):
                config_summary.append("🔹 Data Augmentation: Enabled")
        elif data_type == "text":
            config_summary.append(
                f"🔹 Architecture: {dl_config.get('architecture', 'lstm').upper()}"
            )
            config_summary.append(
                f"🔹 Embedding: {dl_config.get('embedding_dim', 128)}D"
            )
            config_summary.append(f"🔹 Hidden: {dl_config.get('hidden_dim', 128)}D")

        config_summary.append(f"🔹 Batch Size: {dl_config.get('batch_size', 32)}")
        config_summary.append(f"🔹 Max Epochs: {dl_config.get('max_epochs', 50)}")
        config_summary.append(
            f"🔹 Learning Rate: {dl_config.get('learning_rate', 0.001)}"
        )
        config_summary.append(
            f"🔹 Optimizer: {dl_config.get('optimizer_name', 'adam').upper()}"
        )

        st.info("\n\n".join(config_summary))

    st.markdown("---")

    # Train button
    if st.button("🚀 Start Training", type="primary", width="stretch"):
        st.session_state.training_started = True

        try:
            if use_deep_learning:
                # Deep Learning Training
                data_type = st.session_state.get("data_type", "tabular")

                with st.spinner(
                    f"Training {data_type} deep learning model... This may take several minutes."
                ):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Initializing model...")
                    progress_bar.progress(10)

                    if data_type == "tabular":
                        from automl.models.deep_learning.mlp_models import (
                            MLPClassifier,
                            MLPRegressor,
                        )

                        status_text.text("Preparing tabular data...")
                        progress_bar.progress(20)

                        # Determine problem type using smart detection
                        if df is not None and target_column is not None:
                            target_data = df[target_column]
                        else:
                            st.error("Data or target column not available")
                            return

                        if problem_type == "auto":
                            # Smart auto-detection
                            n_unique = target_data.nunique()
                            n_samples = len(target_data)
                            is_numeric = pd.api.types.is_numeric_dtype(target_data)
                            is_categorical = (
                                target_data.dtype == "object"
                                or target_data.dtype.name == "category"
                            )

                            if is_categorical:
                                is_classification = True
                            elif is_numeric:
                                unique_ratio = n_unique / n_samples
                                # Classification if discrete or low cardinality
                                is_classification = (
                                    n_unique <= 20 or unique_ratio < 0.05
                                )
                            else:
                                is_classification = True  # Default to classification

                            detected_type = (
                                "classification" if is_classification else "regression"
                            )
                            st.info(
                                f"🔍 Auto-detected: **{detected_type.upper()}** ({n_unique} unique values)"
                            )
                        else:
                            is_classification = problem_type == "classification"

                        # Get features and target
                        if df is not None and target_column is not None:
                            X = df.drop(columns=[target_column]).values
                            y = df[target_column].values
                        else:
                            st.error("Data or target column not available")
                            return

                        # Train/val/test split (60% train, 20% val, 20% test)
                        from sklearn.model_selection import train_test_split

                        # First split: separate test set
                        X_temp, X_test, y_temp, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                        # Second split: separate train and validation from remaining data
                        val_size = 0.2 / (
                            1 - test_size
                        )  # Proportion of temp data to use for validation
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_temp, y_temp, test_size=val_size, random_state=42
                        )

                        st.info(
                            f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} samples"
                        )

                        progress_bar.progress(40)
                        status_text.text("Training MLP model...")

                        # Create and train model
                        if is_classification:
                            model = MLPClassifier(**dl_config)
                        else:
                            model = MLPRegressor(**dl_config)

                        # Train with validation data
                        model.fit(
                            X_train, y_train, X_val=X_val, y_val=y_val, verbose=True
                        )

                        progress_bar.progress(90)
                        status_text.text("Evaluating model...")

                        # Make predictions
                        train_preds = model.predict(X_train)
                        test_preds = model.predict(X_test)

                        # Calculate metrics
                        if is_classification:
                            from sklearn.metrics import (
                                accuracy_score,
                                classification_report,
                            )

                            train_acc = accuracy_score(y_train, train_preds)
                            test_acc = accuracy_score(y_test, test_preds)

                            # Save model to disk before cleanup
                            os.makedirs("saved_models", exist_ok=True)
                            model_path = f'saved_models/mlp_classifier_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
                            model.save_model(model_path)

                            results = {
                                "model_type": "MLP Classifier",
                                "train_accuracy": train_acc,
                                "test_accuracy": test_acc,
                                "config": dl_config,
                                "history": model.history,
                                "model_path": model_path,
                                "data_type": "tabular",
                            }
                            st.session_state.results = results

                            # Clean up model to free memory
                            del model
                            import gc

                            gc.collect()

                            progress_bar.progress(100)
                            status_text.empty()

                            st.success("✅ Training completed!")
                            st.metric("Test Accuracy", f"{test_acc:.4f}")
                        else:
                            from sklearn.metrics import mean_squared_error, r2_score

                            train_mse = mean_squared_error(y_train, train_preds)
                            test_mse = mean_squared_error(y_test, test_preds)
                            test_r2 = r2_score(y_test, test_preds)

                            # Save model to disk before cleanup
                            os.makedirs("saved_models", exist_ok=True)
                            model_path = f'saved_models/mlp_regressor_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
                            model.save_model(model_path)

                            results = {
                                "model_type": "MLP Regressor",
                                "train_mse": train_mse,
                                "test_mse": test_mse,
                                "test_r2": test_r2,
                                "config": dl_config,
                                "history": model.history,
                                "model_path": model_path,
                                "data_type": "tabular",
                            }
                            st.session_state.results = results

                            # Clean up model to free memory
                            del model
                            import gc

                            gc.collect()

                            progress_bar.progress(100)
                            status_text.empty()

                            st.success("✅ Training completed!")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Test MSE", f"{test_mse:.4f}")
                            with col2:
                                st.metric("Test R²", f"{test_r2:.4f}")

                    elif data_type == "image":
                        from sklearn.preprocessing import LabelEncoder

                        from automl.models.deep_learning.vision.cnn_models import (
                            CNNClassifier,
                        )

                        status_text.text("Preparing image data...")
                        progress_bar.progress(20)

                        # Get image data from session state
                        image_data = st.session_state.get("image_data")

                        if not image_data:
                            st.error(
                                "❌ No image data found. Please upload images first."
                            )
                            return

                        image_files = image_data["files"]
                        labels_str = image_data["labels"]

                        # Convert string labels to integers
                        le = LabelEncoder()
                        labels = le.fit_transform(labels_str)

                        # Get class information
                        class_names = le.classes_
                        num_classes = len(class_names)

                        st.info(
                            f"📊 Training on {len(image_files)} images from {num_classes} classes: {', '.join(class_names)}"
                        )

                        # Split data into train/val (80/20)
                        from sklearn.model_selection import train_test_split

                        X_train, X_val, y_train, y_val = train_test_split(
                            image_files,
                            labels,
                            test_size=0.2,
                            random_state=42,
                            stratify=labels,
                        )

                        st.info(
                            f"📊 Data split: Train={len(X_train)}, Val={len(X_val)} images"
                        )

                        # Check if training multiple architectures
                        train_multiple = st.session_state.get(
                            "train_multiple_dl", False
                        )

                        if train_multiple and "architectures_to_try" in dl_config:
                            # Multi-architecture training (AutoML mode)
                            architectures = dl_config["architectures_to_try"]
                            quick_epochs = dl_config.get("quick_epochs", 15)

                            st.info(
                                f"🔬 Training {len(architectures)} architectures with {quick_epochs} epochs each..."
                            )

                            best_val_acc = 0
                            best_arch = None
                            best_model_path = None
                            all_results = []

                            for i, arch in enumerate(architectures):
                                progress_bar.progress(
                                    20 + (i * 60 // len(architectures))
                                )
                                status_text.text(
                                    f"Training {arch.upper()} ({i+1}/{len(architectures)})..."
                                )

                                # Get model configuration
                                image_size = dl_config.get("image_size", 224)
                                if isinstance(image_size, int):
                                    image_size = (image_size, image_size)
                                batch_size = dl_config.get("batch_size", 32)
                                learning_rate = dl_config.get("learning_rate", 0.001)

                                # Create model
                                model = CNNClassifier(
                                    architecture=arch,
                                    num_classes=num_classes,
                                    image_size=image_size,
                                    batch_size=batch_size,
                                    max_epochs=quick_epochs,
                                    learning_rate=learning_rate,
                                    use_gpu=gpu_available,
                                )

                                # Show training start
                                st.caption(
                                    f"🔄 Starting {arch.upper()} with {quick_epochs} epochs..."
                                )

                                # Train the model with verbose output
                                model.fit(
                                    X_train,
                                    y_train,
                                    X_val=X_val,
                                    y_val=y_val,
                                    verbose=True,
                                )

                                # Get validation accuracy
                                history = (
                                    model.history if hasattr(model, "history") else {}
                                )
                                val_acc = (
                                    history.get("val_metric", [0])[-1]
                                    if history.get("val_metric")
                                    else 0
                                )
                                train_acc = (
                                    history.get("train_metric", [0])[-1]
                                    if history.get("train_metric")
                                    else 0
                                )

                                # Save model
                                os.makedirs("saved_models", exist_ok=True)
                                model_path = f'saved_models/cnn_{arch}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
                                model.save_model(model_path)

                                # Track results
                                all_results.append(
                                    {
                                        "architecture": arch,
                                        "train_accuracy": train_acc,
                                        "val_accuracy": val_acc,
                                        "model_path": model_path,
                                        "history": history,
                                    }
                                )

                                st.caption(
                                    f"✓ {arch.upper()}: Train={train_acc:.2%}, Val={val_acc:.2%}"
                                )

                                # Update best model
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_arch = arch
                                    best_model_path = model_path

                                # Clean up
                                del model
                                import gc

                                gc.collect()

                            # Use best architecture for final results
                            progress_bar.progress(90)
                            status_text.text("Selecting best model...")

                            best_result = next(
                                r for r in all_results if r["architecture"] == best_arch
                            )

                            best_arch_str = (
                                best_arch.upper()
                                if best_arch is not None
                                else "UNKNOWN"
                            )
                            st.success(
                                f"✅ Best Model: **{best_arch_str}** with {best_val_acc:.2%} validation accuracy!"
                            )

                            # Store comprehensive results
                            results = {
                                "model_type": f"CNN {best_arch_str} (Auto-Selected)",
                                "num_classes": num_classes,
                                "class_names": class_names.tolist(),
                                "num_images": len(image_files),
                                "val_accuracy": best_val_acc,
                                "config": dl_config,
                                "history": best_result["history"],
                                "model_path": best_model_path,
                                "data_type": "image",
                                "all_architectures_tested": all_results,
                            }
                            st.session_state.results = results
                            st.session_state.label_encoder = le

                            progress_bar.progress(100)
                            status_text.empty()

                            # Show comparison table
                            st.markdown("### 📊 Architecture Comparison")
                            comparison_df = pd.DataFrame(
                                [
                                    {
                                        "Architecture": r["architecture"].upper(),
                                        "Train Accuracy": f"{r['train_accuracy']:.2%}",
                                        "Val Accuracy": f"{r['val_accuracy']:.2%}",
                                        "Status": (
                                            "🏆 Best"
                                            if r["architecture"] == best_arch
                                            else ""
                                        ),
                                    }
                                    for r in sorted(
                                        all_results,
                                        key=lambda x: x["val_accuracy"],
                                        reverse=True,
                                    )
                                ]
                            )
                            st.dataframe(comparison_df, width="stretch")

                            # Show classes
                            st.markdown("### 📋 Classes")
                            st.write(", ".join(class_names))

                        else:
                            # Single architecture training
                            architecture = dl_config.get("architecture", "resnet18")
                            image_size = dl_config.get("image_size", 224)
                            # Convert to tuple if int
                            if isinstance(image_size, int):
                                image_size = (image_size, image_size)
                            batch_size = dl_config.get("batch_size", 32)
                            max_epochs = dl_config.get("max_epochs", 30)
                            learning_rate = dl_config.get("learning_rate", 0.001)
                            use_gpu = dl_config.get("use_gpu", gpu_available)

                            progress_bar.progress(40)
                            status_text.text(
                                f"Training {architecture.upper()} model..."
                            )

                            # Create and train model
                            model = CNNClassifier(
                                architecture=architecture,
                                num_classes=num_classes,
                                image_size=image_size,
                                batch_size=batch_size,
                                max_epochs=max_epochs,
                                learning_rate=learning_rate,
                                use_gpu=use_gpu,
                            )

                            # Train the model with validation data
                            model.fit(
                                X_train, y_train, X_val=X_val, y_val=y_val, verbose=True
                            )

                            progress_bar.progress(90)
                            status_text.text("Evaluating model...")

                            # Get training history
                            history = model.history if hasattr(model, "history") else {}

                            # Calculate final metrics (CNN stores as 'train_metric' and 'val_metric')
                            final_train_loss = (
                                history.get("train_loss", [0])[-1]
                                if history.get("train_loss")
                                else 0
                            )
                            final_train_acc = (
                                history.get("train_metric", [0])[-1]
                                if history.get("train_metric")
                                else 0
                            )
                            final_val_loss = (
                                history.get("val_loss", [0])[-1]
                                if history.get("val_loss")
                                else 0
                            )
                            final_val_acc = (
                                history.get("val_metric", [0])[-1]
                                if history.get("val_metric")
                                else 0
                            )

                            # Save model to disk before cleanup
                            os.makedirs("saved_models", exist_ok=True)
                            model_path = f'saved_models/cnn_{architecture}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
                            model.save_model(model_path)

                            # Store results
                            results = {
                                "model_type": f"CNN {architecture.upper()}",
                                "num_classes": num_classes,
                                "class_names": class_names.tolist(),
                                "num_images": len(image_files),
                                "train_loss": final_train_loss,
                                "train_accuracy": final_train_acc,
                                "val_loss": final_val_loss,
                                "val_accuracy": final_val_acc,
                                "config": dl_config,
                                "history": history,
                                "model_path": model_path,
                                "data_type": "image",
                            }
                            st.session_state.results = results
                            st.session_state.label_encoder = le

                            # Clean up model to free memory
                            del model
                            import gc

                            gc.collect()

                            progress_bar.progress(100)
                            status_text.empty()

                            st.success("✅ Training completed!")

                            # Show training summary
                            st.markdown("### 📊 Training Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Train Loss", f"{final_train_loss:.4f}")
                            with col2:
                                st.metric("Train Accuracy", f"{final_train_acc:.2%}")
                            with col3:
                                st.metric("Val Loss", f"{final_val_loss:.4f}")
                            with col4:
                                st.metric("Val Accuracy", f"{final_val_acc:.2%}")

                            # Show class distribution
                            st.markdown("### 📋 Classes")
                            st.write(", ".join(class_names))

                        st.info(
                            "👉 Go to 'Results & Comparison' to see detailed training history and analysis"
                        )

                    elif data_type == "text":
                        from automl.models.deep_learning.nlp import TextClassifier

                        status_text.text("Preparing text data...")
                        progress_bar.progress(20)

                        # Get text data
                        text_data = st.session_state.get("text_data")

                        if not text_data:
                            st.error(
                                "❌ No text data found. Please upload text data first."
                            )
                            return

                        # Handle both old format (lists) and new format (dataframe)
                        if text_data.get("is_large", False):
                            # Large dataset - use dataframe
                            df = text_data["dataframe"]
                            text_col = text_data["text_column"]
                            label_col = text_data["label_column"]

                            # For very large datasets, sample for faster training
                            if len(df) > 50000:
                                st.info(
                                    f"💡 Large dataset ({len(df):,} rows). Using 50,000 samples for training."
                                )
                                df = df.sample(50000, random_state=42)

                            texts = df[text_col].tolist()
                            labels = df[label_col].tolist()
                        else:
                            # Small dataset - use lists
                            texts = text_data["texts"]
                            labels = text_data["labels"]

                        st.info(f"📊 Training on {len(texts):,} text samples")

                        # Convert labels to numeric if needed
                        from sklearn.preprocessing import LabelEncoder

                        le = LabelEncoder()
                        labels_numeric = le.fit_transform(labels)

                        # Train/val/test split (60% train, 20% val, 20% test)
                        from sklearn.model_selection import train_test_split

                        # First split: separate test set
                        (
                            texts_temp,
                            test_texts,
                            labels_temp,
                            test_labels,
                        ) = train_test_split(
                            texts, labels_numeric, test_size=test_size, random_state=42
                        )

                        # Second split: separate train and validation from remaining data
                        val_size = 0.2 / (
                            1 - test_size
                        )  # Proportion of temp data to use for validation
                        (
                            train_texts,
                            val_texts,
                            train_labels,
                            val_labels,
                        ) = train_test_split(
                            texts_temp, labels_temp, test_size=val_size, random_state=42
                        )

                        st.info(
                            f"📊 Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)} samples"
                        )

                        progress_bar.progress(40)
                        status_text.text(
                            f"Training {dl_config['architecture'].upper()} model..."
                        )

                        # Create and train model
                        model = TextClassifier(**dl_config)
                        # Train with validation data
                        model.fit(
                            train_texts,
                            train_labels,
                            X_val=val_texts,
                            y_val=val_labels,
                            verbose=True,
                        )

                        progress_bar.progress(90)
                        status_text.text("Evaluating model...")

                        # Make predictions
                        test_preds = model.predict(test_texts)

                        # Calculate accuracy
                        from sklearn.metrics import (
                            accuracy_score,
                            classification_report,
                        )

                        test_acc = accuracy_score(test_labels, test_preds)

                        # Save model to disk before cleanup
                        os.makedirs("saved_models", exist_ok=True)
                        model_path = f'saved_models/nlp_{dl_config["architecture"]}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
                        model.save_model(model_path)

                        results = {
                            "model_type": f'NLP {dl_config["architecture"].upper()}',
                            "test_accuracy": test_acc,
                            "config": dl_config,
                            "history": model.history,
                            "model_path": model_path,
                            "data_type": "text",
                        }
                        st.session_state.results = results

                        progress_bar.progress(100)
                        status_text.empty()

                        st.success("✅ Training completed!")
                        st.metric("Test Accuracy", f"{test_acc:.4f}")

                        # Show sample predictions
                        st.markdown("#### Sample Predictions")
                        sample_texts = test_texts[:5]
                        sample_true = [
                            le.inverse_transform([l])[0] for l in test_labels[:5]
                        ]
                        sample_preds = model.predict(sample_texts)
                        sample_pred_labels = [
                            le.inverse_transform([p])[0] for p in sample_preds
                        ]

                        pred_df = pd.DataFrame(
                            {
                                "Text": [
                                    t[:100] + "..." if len(t) > 100 else t
                                    for t in sample_texts
                                ],
                                "True Label": sample_true,
                                "Predicted": sample_pred_labels,
                            }
                        )
                        st.dataframe(pred_df)

                        # Clean up model to free memory (after showing predictions)
                        del model
                        del le
                        import gc

                        gc.collect()

                st.info("👉 Go to 'Results & Comparison' to see detailed analysis")

            else:
                # Traditional ML Training (existing code)
                with st.spinner("Training models... This may take a few minutes."):
                    # Import AutoML
                    from automl.pipeline.automl import AutoML

                # Initialize AutoML
                automl = AutoML(
                    problem_type=None if problem_type == "auto" else problem_type,
                    use_cross_validation=use_cv,
                    cv_folds=cv_folds,
                    test_size=test_size,
                    validation_size=0.2,
                    verbose=False,
                    optimize_hyperparameters=optimize_hp,
                    n_trials=n_trials,
                    enable_mlflow=enable_mlflow,
                )

                # Fit the model
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Loading and validating data...")
                progress_bar.progress(15)

                status_text.text("Performing EDA...")
                progress_bar.progress(30)

                status_text.text("Preprocessing data...")
                progress_bar.progress(45)

                status_text.text("Training models...")

                # Get selected models from session state
                models_to_train = st.session_state.get("models_to_train", None)

                # Handle "all" marker
                if models_to_train == "all":
                    from automl.models.model_metadata import list_models_by_type

                    # Get actual problem type from automl after initialization
                    detected_problem_type = (
                        automl.problem_type
                        if hasattr(automl, "problem_type") and automl.problem_type
                        else problem_type
                    )
                    if detected_problem_type == "auto":
                        # Quick detection
                        if df is not None and target_column is not None:
                            target_data = df[target_column]
                            n_unique = target_data.nunique()
                            is_numeric = pd.api.types.is_numeric_dtype(target_data)
                            detected_problem_type = (
                                "regression"
                                if (is_numeric and n_unique > 20)
                                else "classification"
                            )
                        else:
                            detected_problem_type = "classification"  # Default fallback

                    models_to_train = list_models_by_type(detected_problem_type)
                    st.info(
                        f"🌟 Training **all {len(models_to_train)} models** for {detected_problem_type}"
                    )

                # Fit with selected models
                if df is not None and target_column is not None:
                    results = automl.fit(
                        df, target_column=target_column, models_to_try=models_to_train
                    )
                else:
                    st.error("Data or target column not available")
                    return

                # Save the best model for download before cleanup
                best_model_bytes = None
                mlflow_run_id = None
                try:
                    if hasattr(automl, "best_model") and automl.best_model:
                        import pickle

                        best_model_bytes = pickle.dumps(automl.best_model)
                    # Also save MLflow run ID for accessing artifacts
                    tracker = getattr(automl, "tracker", None)
                    if tracker and hasattr(tracker, "run_id"):
                        mlflow_run_id = tracker.run_id
                except Exception as e:
                    st.warning(f"Could not save model for download: {str(e)}")

                # Delete the automl object immediately to free memory
                # (it contains large trained models that can crash Streamlit)
                del automl

                progress_bar.progress(95)

                status_text.text("Finalizing results...")
                progress_bar.progress(100)

                # Store results (avoid storing large objects)
                # Extract only serializable, essential data
                try:
                    # Get model comparison data without trained model objects
                    model_comparison = results.get("model_comparison", {})

                    # Remove trained_models from comparison (they're not serializable)
                    safe_comparison = {
                        "models": model_comparison.get("models", []),
                        "model_names": model_comparison.get("model_names", []),
                        "rankings": model_comparison.get("rankings", []),
                        "best_score": model_comparison.get("best_score", 0),
                    }

                    # Clean model results - remove model objects
                    if "models" in safe_comparison:
                        for model_result in safe_comparison["models"]:
                            # Remove any non-serializable objects
                            if "model" in model_result:
                                del model_result["model"]

                    st.session_state.results = {
                        "problem_type": results.get("problem_type", "unknown"),
                        "n_features": results.get("n_features", 0),
                        "n_samples_train": results.get("n_samples_train", 0),
                        "model_comparison": safe_comparison,
                        "best_model": results.get("best_model", "N/A"),
                        "best_model_bytes": best_model_bytes,  # For download
                        "mlflow_run_id": mlflow_run_id,  # For accessing artifacts
                    }
                    st.session_state.training_complete = True

                    # Clean up results object
                    del results

                    # Force garbage collection to free memory immediately
                    import gc

                    gc.collect()

                except Exception as e:
                    st.error(f"Error storing results: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

                status_text.empty()
                progress_bar.empty()

                st.success("✅ Training completed successfully!")

                # Show quick summary
                st.markdown("### Quick Summary")

                try:
                    col1, col2, col3 = st.columns(3)

                    # Extract model comparison data safely from session state
                    stored_results = st.session_state.results
                    model_comparison = stored_results.get("model_comparison", {})
                    rankings = model_comparison.get("rankings", [])
                    best_score = model_comparison.get("best_score", 0)
                    problem_type = stored_results.get("problem_type", "classification")

                    # Get accuracy from best model if available
                    best_model_accuracy = best_score  # default to best_score
                    metric_label = "Best Score"

                    if problem_type == "classification":
                        # Try to get accuracy specifically from best model's metrics
                        all_model_results = model_comparison.get("models", [])
                        best_model_name = stored_results.get("best_model", "")

                        for model_result in all_model_results:
                            if (
                                model_result.get("model_name") == best_model_name
                                and model_result.get("status") == "success"
                            ):
                                # Check cross-validation results first
                                if "cross_validation" in model_result:
                                    cv_results = model_result["cross_validation"]
                                    best_model_accuracy = cv_results.get(
                                        "mean_accuracy", best_score
                                    )
                                # Then validation metrics
                                elif "val_metrics" in model_result:
                                    best_model_accuracy = model_result[
                                        "val_metrics"
                                    ].get("accuracy", best_score)
                                # Finally training metrics
                                elif "train_metrics" in model_result:
                                    best_model_accuracy = model_result[
                                        "train_metrics"
                                    ].get("accuracy", best_score)
                                break

                        metric_label = "Accuracy"
                    elif problem_type == "regression":
                        metric_label = "R² Score"

                    with col1:
                        st.metric("Best Model", stored_results.get("best_model", "N/A"))

                    with col2:
                        st.metric(metric_label, f"{best_model_accuracy:.4f}")

                    with col3:
                        st.metric("Models Trained", len(rankings) if rankings else 0)

                    st.info("👉 Go to 'Results & Comparison' to see detailed analysis")

                except Exception as e:
                    st.warning(
                        f"Training completed but could not display summary: {str(e)}"
                    )
                    st.info("👉 Go to 'Results & Comparison' to see detailed analysis")

        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.exception(e)


def generate_pdf_report(results, filename="automl_report.pdf"):
    """
    Generate a comprehensive PDF report from results.

    Args:
        results: Results dictionary
        filename: Output filename

    Returns:
        BytesIO object containing the PDF
    """
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Overview
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        # Title
        fig.text(
            0.5,
            0.95,
            "AutoML Results Report",
            ha="center",
            va="top",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.90,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
        )

        # Check if deep learning or traditional ML
        is_deep_learning = "model_type" in results and any(
            x in results.get("model_type", "")
            for x in ["MLP", "CNN", "NLP", "LSTM", "GRU"]
        )

        if is_deep_learning:
            # Deep Learning Report
            y_pos = 0.80

            # Model Info
            fig.text(0.1, y_pos, "Model Information", fontsize=16, fontweight="bold")
            y_pos -= 0.05
            fig.text(
                0.1,
                y_pos,
                f"Model Type: {results.get('model_type', 'N/A')}",
                fontsize=12,
            )
            y_pos -= 0.04
            fig.text(
                0.1,
                y_pos,
                f"Data Type: {results.get('data_type', 'N/A').title()}",
                fontsize=12,
            )
            y_pos -= 0.04

            # Performance Metrics
            y_pos -= 0.05
            fig.text(0.1, y_pos, "Performance Metrics", fontsize=16, fontweight="bold")
            y_pos -= 0.05

            if "val_accuracy" in results:
                fig.text(
                    0.1,
                    y_pos,
                    f"Validation Accuracy: {results['val_accuracy']:.2%}",
                    fontsize=12,
                )
                y_pos -= 0.04
            if "train_accuracy" in results:
                fig.text(
                    0.1,
                    y_pos,
                    f"Training Accuracy: {results['train_accuracy']:.2%}",
                    fontsize=12,
                )
                y_pos -= 0.04
            if "test_accuracy" in results:
                fig.text(
                    0.1,
                    y_pos,
                    f"Test Accuracy: {results['test_accuracy']:.2%}",
                    fontsize=12,
                )
                y_pos -= 0.04

            # Configuration
            if "config" in results:
                y_pos -= 0.05
                fig.text(
                    0.1, y_pos, "Model Configuration", fontsize=16, fontweight="bold"
                )
                y_pos -= 0.05
                config = results["config"]
                for key, value in list(config.items())[:8]:  # Limit to avoid overflow
                    fig.text(0.1, y_pos, f"{key}: {value}", fontsize=10)
                    y_pos -= 0.03
        else:
            # Traditional ML Report
            y_pos = 0.80

            # Best Model Info
            fig.text(0.1, y_pos, "Best Model", fontsize=16, fontweight="bold")
            y_pos -= 0.05
            fig.text(
                0.1, y_pos, f"Model: {results.get('best_model', 'N/A')}", fontsize=12
            )
            y_pos -= 0.04
            fig.text(
                0.1,
                y_pos,
                f"Problem Type: {results.get('problem_type', 'N/A').title()}",
                fontsize=12,
            )
            y_pos -= 0.04

            # Model Comparison
            model_comparison = results.get("model_comparison", {})
            rankings = model_comparison.get("rankings", [])
            best_score = model_comparison.get("best_score", 0)

            y_pos -= 0.05
            fig.text(0.1, y_pos, "Performance", fontsize=16, fontweight="bold")
            y_pos -= 0.05
            fig.text(0.1, y_pos, f"Best Score: {best_score:.4f}", fontsize=12)
            y_pos -= 0.04
            fig.text(0.1, y_pos, f"Models Tested: {len(rankings)}", fontsize=12)
            y_pos -= 0.04

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: Model Comparison Chart (if traditional ML)
        if not is_deep_learning and rankings:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

            # Performance comparison
            df = pd.DataFrame(rankings)
            import matplotlib.cm as cm

            colors = cm.viridis(np.linspace(0, 1, len(df)))  # type: ignore[attr-defined]

            ax1.barh(df["model_name"], df["score"], color=colors)
            ax1.set_xlabel("Score", fontsize=12)
            ax1.set_title("Model Accuracy Comparison", fontsize=14)
            ax1.grid(axis="x", alpha=0.3)

            # Training time comparison
            ax2.barh(df["model_name"], df["training_time"], color=colors)
            ax2.set_xlabel("Training Time (seconds)", fontsize=12)
            ax2.set_title("Training Time Comparison", fontsize=14)
            ax2.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Page 3: Training History (if deep learning)
        if is_deep_learning and "history" in results and results["history"]:
            history = results["history"]

            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle("Training History", fontsize=16, fontweight="bold")

            # Loss curves
            if "train_loss" in history:
                epochs = range(1, len(history["train_loss"]) + 1)
                axes[0].plot(
                    epochs,
                    history["train_loss"],
                    "b-o",
                    label="Train Loss",
                    linewidth=2,
                )
                if "val_loss" in history:
                    axes[0].plot(
                        epochs,
                        history["val_loss"],
                        "r--s",
                        label="Val Loss",
                        linewidth=2,
                    )
                axes[0].set_xlabel("Epoch", fontsize=12)
                axes[0].set_ylabel("Loss", fontsize=12)
                axes[0].set_title("Loss Curves", fontsize=14)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Accuracy curves
            train_acc_key = next(
                (
                    k
                    for k in ["train_accuracy", "train_acc", "train_metric"]
                    if k in history and history.get(k)
                ),
                None,
            )
            val_acc_key = next(
                (
                    k
                    for k in ["val_accuracy", "val_acc", "val_metric"]
                    if k in history and history.get(k)
                ),
                None,
            )

            if train_acc_key:
                epochs = range(1, len(history[train_acc_key]) + 1)
                axes[1].plot(
                    epochs,
                    history[train_acc_key],
                    "g-o",
                    label="Train Acc",
                    linewidth=2,
                )
                if val_acc_key:
                    axes[1].plot(
                        epochs,
                        history[val_acc_key],
                        "m--s",
                        label="Val Acc",
                        linewidth=2,
                    )
                axes[1].set_xlabel("Epoch", fontsize=12)
                axes[1].set_ylabel("Accuracy", fontsize=12)
                axes[1].set_title("Accuracy Curves", fontsize=14)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    buffer.seek(0)
    return buffer


def create_comparison_charts(rankings):
    """
    Create comprehensive comparison charts for multiple models.

    Args:
        rankings: List of model ranking dictionaries

    Returns:
        List of plotly figures
    """
    if not rankings:
        return []

    df = pd.DataFrame(rankings)
    figures = []

    # 1. Scatter plot: Score vs Training Time
    fig_scatter = px.scatter(
        df,
        x="training_time",
        y="score",
        text="model_name",
        size="score",
        color="score",
        title="Model Performance vs Training Time",
        labels={"training_time": "Training Time (s)", "score": "Score"},
        color_continuous_scale="viridis",
    )

    fig_scatter.update_traces(textposition="top center", textfont_size=10)
    fig_scatter.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    figures.append(("scatter", fig_scatter))

    # 2. Radar chart (if we have at least 3 models)
    if len(df) >= 3:
        # Normalize metrics for radar chart
        df_norm = df.copy()
        df_norm["norm_score"] = (df["score"] - df["score"].min()) / (
            df["score"].max() - df["score"].min() + 1e-10
        )
        # Inverse normalize time (lower is better)
        df_norm["norm_time"] = 1 - (df["training_time"] - df["training_time"].min()) / (
            df["training_time"].max() - df["training_time"].min() + 1e-10
        )

        fig_radar = go.Figure()

        for idx, row in df_norm.head(5).iterrows():  # Top 5 models
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row["norm_score"], row["norm_time"]],
                    theta=["Performance", "Speed"],
                    fill="toself",
                    name=row["model_name"],
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Characteristics (Normalized)",
            height=500,
        )
        figures.append(("radar", fig_radar))

    # 3. Grouped bar chart for side-by-side comparison
    fig_grouped = go.Figure()

    # Normalize values for comparison
    max_score = df["score"].max()
    max_time = df["training_time"].max()

    fig_grouped.add_trace(
        go.Bar(
            name="Performance",
            x=df["model_name"],
            y=df["score"],
            text=[f"{v:.3f}" for v in df["score"]],
            textposition="outside",
            marker_color="rgb(102, 126, 234)",
        )
    )

    fig_grouped.add_trace(
        go.Bar(
            name="Training Time (s)",
            x=df["model_name"],
            y=df["training_time"],
            text=[f"{v:.2f}s" for v in df["training_time"]],
            textposition="outside",
            marker_color="rgb(245, 87, 108)",
            yaxis="y2",
        )
    )

    fig_grouped.update_layout(
        title="Model Comparison: Performance & Training Time",
        xaxis=dict(title="Model"),
        yaxis=dict(title="Score", side="left"),
        yaxis2=dict(title="Training Time (s)", overlaying="y", side="right"),
        barmode="group",
        height=500,
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    figures.append(("grouped", fig_grouped))

    return figures


def results_page():
    """Results visualization and model comparison page."""

    # Animated header
    st.markdown(
        """
    <div class='main-header'>
        <h1 style='margin: 0; font-size: 2.5rem;'>📈 Results & Model Comparison</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Analyze your model performance and compare results</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.results is None:
        st.markdown(
            """
        <div class='info-box'>
            <h3>⚠️ No Results Yet</h3>
            <p>Please train models first to see results and comparisons.</p>
            <p>Go to <strong>🎯 Train Models</strong> to get started!</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    results = st.session_state.results
    # Automl object may not be stored (to save memory)
    automl = st.session_state.get("automl", None)

    # Check if it's a deep learning result
    is_deep_learning = "model_type" in results and any(
        x in results.get("model_type", "") for x in ["MLP", "CNN", "NLP", "LSTM", "GRU"]
    )

    if is_deep_learning:
        # Deep Learning Results View with enhanced visuals
        st.markdown(
            f"""
        <div class='success-box' style='text-align: center;'>
            <h2 style='margin: 0; font-size: 2rem;'>🧠 {results.get('model_type', 'Deep Learning Model')}</h2>
            <p style='margin: 0.5rem 0 0 0;'>Training completed successfully</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📊 Performance Metrics")

        # Enhanced metrics display with gradient cards
        cols = st.columns(4)

        config = results.get("config", {})
        data_type = results.get("data_type", "tabular")

        with cols[0]:
            st.markdown(
                """
            <div class='stat-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Model Type</div>
                <div class='stat-value' style='color: #fee140 !important; -webkit-text-fill-color: #fee140 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{}</div>
            </div>
            """.format(results.get("model_type", "N/A")),
                unsafe_allow_html=True,
            )

        # Display different metrics based on data type
        if "val_accuracy" in results:  # Image/Text models
            with cols[1]:
                acc_color = "#5ee7df" if results["val_accuracy"] > 0.8 else "#f093fb"
                st.markdown(
                    """
                <div class='stat-card' style='background: linear-gradient(135deg, {} 0%, #b490ca 100%);'>
                    <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Val Accuracy</div>
                    <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.2%}</div>
                </div>
                """.format(acc_color, results["val_accuracy"]),
                    unsafe_allow_html=True,
                )

            with cols[2]:
                if "train_accuracy" in results:
                    st.markdown(
                        """
                    <div class='stat-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Train Accuracy</div>
                        <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.2%}</div>
                    </div>
                    """.format(results["train_accuracy"]),
                        unsafe_allow_html=True,
                    )
        elif "test_accuracy" in results:  # MLP models
            with cols[1]:
                acc_color = "#5ee7df" if results["test_accuracy"] > 0.8 else "#f093fb"
                st.markdown(
                    """
                <div class='stat-card' style='background: linear-gradient(135deg, {} 0%, #b490ca 100%);'>
                    <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Test Accuracy</div>
                    <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.2%}</div>
                </div>
                """.format(acc_color, results["test_accuracy"]),
                    unsafe_allow_html=True,
                )

            with cols[2]:
                if "train_accuracy" in results:
                    st.markdown(
                        """
                    <div class='stat-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Train Accuracy</div>
                        <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.2%}</div>
                    </div>
                    """.format(results["train_accuracy"]),
                        unsafe_allow_html=True,
                    )
        elif "test_mse" in results:
            with cols[1]:
                st.markdown(
                    """
                <div class='stat-card' style='background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%);'>
                    <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Test MSE</div>
                    <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.4f}</div>
                </div>
                """.format(results["test_mse"]),
                    unsafe_allow_html=True,
                )

            with cols[2]:
                st.markdown(
                    """
                <div class='stat-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                    <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Test R²</div>
                    <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{:.4f}</div>
                </div>
                """.format(results.get("test_r2", 0)),
                    unsafe_allow_html=True,
                )

        with cols[3]:
            st.markdown(
                """
            <div class='stat-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
                <div class='stat-label' style='color: rgba(255,255,255,0.9);'>Epochs</div>
                <div class='stat-value' style='color: white !important; -webkit-text-fill-color: white !important;'>{}</div>
            </div>
            """.format(config.get("max_epochs", "N/A")),
                unsafe_allow_html=True,
            )

        # Show data-specific information
        if data_type == "image":
            st.markdown("### 🖼️ Dataset Information")
            col_ds1, col_ds2, col_ds3 = st.columns(3)

            with col_ds1:
                st.metric("📸 Total Images", f"{results.get('num_images', 'N/A'):,}")
            with col_ds2:
                st.metric("🏷️ Number of Classes", results.get("num_classes", "N/A"))
            with col_ds3:
                if "class_names" in results:
                    st.metric("📋 Classes", "View below ↓")

            # Show class names
            if "class_names" in results:
                class_names = results["class_names"]
                st.markdown("**Class Names:**")
                # Display in a nice grid
                cols_per_row = 5
                for i in range(0, len(class_names), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(class_names):
                            col.markdown(f"`{class_names[i + j]}`")

        st.markdown("---")

        # Training History with enhanced visualization
        if "history" in results and results["history"]:
            st.markdown("### 📈 Training Progress")

            history = results["history"]

            # Debug info (collapsible)
            with st.expander("🔍 View Available Metrics", expanded=False):
                st.caption("Metrics tracked during training:")
                available_metrics = []
                for key, value in history.items():
                    if isinstance(value, list):
                        available_metrics.append(f"**{key}**: {len(value)} epochs")
                st.markdown(
                    "  \n".join(available_metrics)
                    if available_metrics
                    else "No metrics found"
                )

            # Check what metrics we have
            has_loss = "train_loss" in history and history.get("train_loss")
            has_val_loss = "val_loss" in history and history.get("val_loss")
            has_accuracy = (
                ("train_accuracy" in history and history.get("train_accuracy"))
                or ("train_acc" in history and history.get("train_acc"))
                or ("train_metric" in history and history.get("train_metric"))
            )
            has_val_accuracy = (
                ("val_accuracy" in history and history.get("val_accuracy"))
                or ("val_acc" in history and history.get("val_acc"))
                or ("val_metric" in history and history.get("val_metric"))
            )

            # Create tabs for different metrics
            tab1, tab2 = st.tabs(["📉 Loss Curves", "📊 Accuracy Curves"])

            with tab1:
                # Create enhanced training plots
                if has_loss or has_val_loss:
                    fig_loss = go.Figure()

                    if has_loss:
                        fig_loss.add_trace(
                            go.Scatter(
                                y=history["train_loss"],
                                mode="lines+markers",
                                name="Train Loss",
                                line=dict(color="#667eea", width=3),
                                marker=dict(size=6, symbol="circle"),
                                fill="tozeroy",
                                fillcolor="rgba(102, 126, 234, 0.1)",
                            )
                        )

                    if has_val_loss:
                        fig_loss.add_trace(
                            go.Scatter(
                                y=history["val_loss"],
                                mode="lines+markers",
                                name="Validation Loss",
                                line=dict(color="#f5576c", width=3, dash="dash"),
                                marker=dict(size=6, symbol="diamond"),
                            )
                        )

                    fig_loss.update_layout(
                        title={
                            "text": "Training & Validation Loss",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"size": 20, "color": "#667eea"},
                        },
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        hovermode="x unified",
                        height=500,
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(248,249,250,1)",
                        font=dict(size=12),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                    fig_loss.update_xaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )
                    fig_loss.update_yaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )

                    st.plotly_chart(fig_loss, width="stretch")

                    # Add min/max loss info
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        if has_loss and history["train_loss"]:
                            min_train_loss = min(history["train_loss"])
                            st.info(f"📉 **Min Train Loss:** {min_train_loss:.4f}")
                    with col_info2:
                        if has_val_loss and history["val_loss"]:
                            min_val_loss = min(history["val_loss"])
                            st.info(f"📉 **Min Val Loss:** {min_val_loss:.4f}")
                        elif not has_val_loss:
                            st.caption(
                                "ℹ️ No validation loss (training without validation split)"
                            )
                else:
                    st.info("📉 No loss metrics found in training history.")

            with tab2:
                if has_accuracy or has_val_accuracy:
                    fig_acc = go.Figure()

                    # Determine which key to use for accuracy metrics
                    train_acc_key = None
                    if "train_accuracy" in history and history.get("train_accuracy"):
                        train_acc_key = "train_accuracy"
                    elif "train_acc" in history and history.get("train_acc"):
                        train_acc_key = "train_acc"
                    elif "train_metric" in history and history.get("train_metric"):
                        train_acc_key = "train_metric"

                    val_acc_key = None
                    if "val_accuracy" in history and history.get("val_accuracy"):
                        val_acc_key = "val_accuracy"
                    elif "val_acc" in history and history.get("val_acc"):
                        val_acc_key = "val_acc"
                    elif "val_metric" in history and history.get("val_metric"):
                        val_acc_key = "val_metric"

                    if train_acc_key:
                        fig_acc.add_trace(
                            go.Scatter(
                                y=history[train_acc_key],
                                mode="lines+markers",
                                name="Train Accuracy",
                                line=dict(color="#5ee7df", width=3),
                                marker=dict(size=6, symbol="circle"),
                                fill="tozeroy",
                                fillcolor="rgba(94, 231, 223, 0.1)",
                            )
                        )

                    if val_acc_key:
                        fig_acc.add_trace(
                            go.Scatter(
                                y=history[val_acc_key],
                                mode="lines+markers",
                                name="Validation Accuracy",
                                line=dict(color="#fa709a", width=3, dash="dash"),
                                marker=dict(size=6, symbol="diamond"),
                            )
                        )

                    fig_acc.update_layout(
                        title={
                            "text": "Training & Validation Accuracy",
                            "y": 0.95,
                            "x": 0.5,
                            "xanchor": "center",
                            "yanchor": "top",
                            "font": {"size": 20, "color": "#5ee7df"},
                        },
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy",
                        hovermode="x unified",
                        height=500,
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(248,249,250,1)",
                        font=dict(size=12),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                    fig_acc.update_xaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )
                    fig_acc.update_yaxes(
                        showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"
                    )

                    st.plotly_chart(fig_acc, width="stretch")

                    # Add max accuracy info
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        if train_acc_key and history.get(train_acc_key):
                            max_train_acc = max(history[train_acc_key])
                            st.success(
                                f"🎯 **Max Train Accuracy:** {max_train_acc:.2%}"
                            )
                    with col_info2:
                        if val_acc_key and history.get(val_acc_key):
                            max_val_acc = max(history[val_acc_key])
                            st.success(f"🎯 **Max Val Accuracy:** {max_val_acc:.2%}")
                else:
                    st.info("📊 No accuracy metrics found in training history.")
                    st.caption(
                        "💡 Available metrics: "
                        + ", ".join(
                            [
                                k
                                for k in history.keys()
                                if k not in ["train_loss", "val_loss"]
                            ]
                        )
                    )

        # Configuration Summary
        st.markdown("### ⚙️ Model Configuration")
        config = results.get("config", {})

        config_df = pd.DataFrame(
            [{"Parameter": k, "Value": str(v)} for k, v in config.items()]
        )

        st.dataframe(config_df, width="stretch")

        # Model Export
        st.markdown("### 💾 Export Model")

        # Check if model path is available (DL models) or model object (traditional ML)
        model = results.get("model")
        model_path = results.get("model_path")
        data_type = results.get("data_type", "tabular")

        if model_path and os.path.exists(model_path):
            # Deep learning model saved to disk
            st.info(f"ℹ️ Model saved to: `{model_path}`")

            with open(model_path, "rb") as f:
                model_data = f.read()

            st.download_button(
                label="⬇️ Download Deep Learning Model (.pth)",
                data=model_data,
                file_name=os.path.basename(model_path),
                mime="application/octet-stream",
                help="PyTorch model checkpoint file",
            )

            with st.expander("📋 Model Configuration"):
                config = results.get("config", {})
                for key, value in config.items():
                    st.text(f"• {key}: {value}")

        elif model:
            # Traditional ML model in memory
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                model.save_model(tmp.name)
                tmp.seek(0)

                st.download_button(
                    label="⬇️ Download ML Model (.pkl)",
                    data=open(tmp.name, "rb").read(),
                    file_name=f"{results.get('model_type', 'model').replace(' ', '_')}.pkl",
                    mime="application/octet-stream",
                    help="Scikit-learn model pickle file",
                )
        else:
            st.warning("⚠️ Model not available for download.")

        return

    # Traditional ML Results View (existing code)
    # Extract model comparison data
    model_comparison = results.get("model_comparison", {})
    rankings = model_comparison.get("rankings", [])
    best_score = model_comparison.get("best_score", 0)
    problem_type = results.get("problem_type", "classification")

    # Get accuracy from best model if available
    best_model_accuracy = best_score  # default to best_score
    metric_label = "Best Score"

    if problem_type == "classification":
        # Try to get accuracy specifically from best model's metrics
        all_model_results = model_comparison.get("models", [])
        best_model_name = results.get("best_model", "")

        for model_result in all_model_results:
            if (
                model_result.get("model_name") == best_model_name
                and model_result.get("status") == "success"
            ):
                # Check cross-validation results first
                if "cross_validation" in model_result:
                    cv_results = model_result["cross_validation"]
                    best_model_accuracy = cv_results.get("mean_accuracy", best_score)
                # Then validation metrics
                elif "val_metrics" in model_result:
                    best_model_accuracy = model_result["val_metrics"].get(
                        "accuracy", best_score
                    )
                # Finally training metrics
                elif "train_metrics" in model_result:
                    best_model_accuracy = model_result["train_metrics"].get(
                        "accuracy", best_score
                    )
                break

        metric_label = "Accuracy"
    elif problem_type == "regression":
        metric_label = "R² Score"

    # Enhanced overview metrics with gradient cards
    st.markdown("### 🏆 Best Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        <div class='stat-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <div class='stat-label'>Best Model</div>
            <div class='stat-value' style='font-size: 1.2rem;'>{}</div>
        </div>
        """.format(results.get("best_model", "N/A")),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class='stat-card' style='background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%);'>
            <div class='stat-label'>{}</div>
            <div class='stat-value'>{:.4f}</div>
        </div>
        """.format(metric_label, best_model_accuracy),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class='stat-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
            <div class='stat-label'>Problem Type</div>
            <div class='stat-value' style='font-size: 1.2rem;'>{}</div>
        </div>
        """.format(results.get("problem_type", "N/A").upper()),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
        <div class='stat-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
            <div class='stat-label'>Models Tested</div>
            <div class='stat-value'>{}</div>
        </div>
        """.format(len(rankings)),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Tabs for results
    tab1, tab2, tab3 = st.tabs(
        ["📊 Model Comparison", "📈 Performance Metrics", "💾 Export Results"]
    )

    with tab1:
        st.markdown("### Model Rankings")

        if rankings:
            # Create ranking dataframe
            ranking_df = pd.DataFrame(rankings)

            # Display as table
            st.dataframe(
                ranking_df.style.format({"score": "{:.4f}", "training_time": "{:.2f}"}),
                width="stretch",
            )

            # Enhanced bar chart comparison with gradient colors
            fig = px.bar(
                ranking_df,
                x="model_name",
                y="score",
                title="Model Performance Comparison",
                labels={"score": "Score", "model_name": "Model"},
                color="score",
                color_continuous_scale=["#667eea", "#5ee7df", "#f5576c"],
            )

            # Enhance the layout
            fig.update_traces(
                marker=dict(line=dict(color="white", width=2)),
                texttemplate="%{y:.4f}",
                textposition="outside",
            )

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                height=500,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    title_font=dict(size=14, color="#666"),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(200,200,200,0.2)",
                    title_font=dict(size=14, color="#666"),
                ),
                title=dict(font=dict(size=18, color="#333"), x=0.5, xanchor="center"),
            )

            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            # Enhanced training time comparison
            fig_time = px.bar(
                ranking_df,
                x="model_name",
                y="training_time",
                title="Training Time Comparison",
                labels={"training_time": "Time (seconds)", "model_name": "Model"},
                color="training_time",
                color_continuous_scale=["#f093fb", "#f5576c", "#fa709a"],
            )

            fig_time.update_traces(
                marker=dict(line=dict(color="white", width=2)),
                texttemplate="%{y:.2f}s",
                textposition="outside",
            )

            fig_time.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                height=500,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    title_font=dict(size=14, color="#666"),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(200,200,200,0.2)",
                    title_font=dict(size=14, color="#666"),
                ),
                title=dict(font=dict(size=18, color="#333"), x=0.5, xanchor="center"),
            )

            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_time, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

            # Add comprehensive comparison charts
            st.markdown("---")
            st.markdown("### 📊 Advanced Model Comparisons")

            comparison_figs = create_comparison_charts(rankings)

            if comparison_figs:
                comp_tab1, comp_tab2, comp_tab3 = st.tabs(
                    [
                        "⚡ Performance vs Speed",
                        "🎯 Model Characteristics",
                        "📊 Dual Comparison",
                    ]
                )

                with comp_tab1:
                    scatter_fig = next(
                        (fig for name, fig in comparison_figs if name == "scatter"),
                        None,
                    )
                    if scatter_fig:
                        st.plotly_chart(scatter_fig, width="stretch")
                        st.caption(
                            "💡 **Interpretation**: Models in the top-left corner offer the bestbalance of high performance and fast training."
                        )

                with comp_tab2:
                    radar_fig = next(
                        (fig for name, fig in comparison_figs if name == "radar"), None
                    )
                    if radar_fig:
                        st.plotly_chart(radar_fig, width="stretch")
                        st.caption(
                            "💡 **Interpretation**: Larger areas indicate better overall performance across metrics."
                        )
                    else:
                        st.info(
                            "📊 Radar chart requires at least 3 models for comparison."
                        )

                with comp_tab3:
                    grouped_fig = next(
                        (fig for name, fig in comparison_figs if name == "grouped"),
                        None,
                    )
                    if grouped_fig:
                        st.plotly_chart(grouped_fig, width="stretch")
                        st.caption(
                            "💡 **Interpretation**: Blue bars show performance (higher is better), red bars show training time."
                        )

    with tab2:
        st.markdown("### Detailed Metrics")

        # Get metrics for best model from the full model results array
        best_model_metrics = {}
        best_model_name = results.get("best_model", "")

        # Extract from model_comparison -> models array
        model_comparison = results.get("model_comparison", {})
        all_model_results = model_comparison.get("models", [])

        # Find the best model's full result
        best_model_result = None
        for model_result in all_model_results:
            if (
                model_result.get("model_name") == best_model_name
                and model_result.get("status") == "success"
            ):
                best_model_result = model_result
                break

        # Extract metrics based on validation type
        if best_model_result:
            if "cross_validation" in best_model_result:
                # Cross-validation metrics - these have mean_ and std_ prefixes
                cv_results = best_model_result["cross_validation"]
                # Extract all mean metrics (they're at the top level of cv_results)
                best_model_metrics = {
                    k: v
                    for k, v in cv_results.items()
                    if k.startswith("mean_")
                    or k in ["mean_primary_metric", "std_primary_metric"]
                }
                # Add std metrics for reference
                std_metrics = {
                    k: v for k, v in cv_results.items() if k.startswith("std_")
                }
                best_model_metrics.update(std_metrics)
            elif "val_metrics" in best_model_result:
                # Validation split metrics
                best_model_metrics = best_model_result["val_metrics"]
            elif "train_metrics" in best_model_result:
                # Fallback to training metrics
                best_model_metrics = best_model_result["train_metrics"]

        if best_model_metrics:
            st.success(
                f"📊 Showing metrics for: **{results.get('best_model', 'Best Model')}**"
            )

            # Check if cross-validation was used
            is_cv = best_model_result and "cross_validation" in best_model_result
            if is_cv and best_model_result is not None:
                st.info(
                    f"📊 Cross-Validation Results ({best_model_result['cross_validation'].get('n_splits', 5)} folds)"
                )

            # Display metrics in organized sections
            # Separate regression and classification metrics (handle mean_ prefix)
            regression_metrics = ["r2", "mse", "rmse", "mae", "mape", "median_ae"]
            classification_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "log_loss",
            ]

            reg_metrics = {}
            class_metrics = {}
            other_metrics = {}

            for k, v in best_model_metrics.items():
                # Skip std metrics for now (we'll display them with means)
                if k.startswith("std_"):
                    continue

                # Remove mean_ prefix for categorization
                clean_key = k.replace("mean_", "")

                if any(rm in clean_key.lower() for rm in regression_metrics):
                    reg_metrics[k] = v
                elif any(cm in clean_key.lower() for cm in classification_metrics):
                    class_metrics[k] = v
                elif k not in [
                    "mean_primary_metric",
                    "std_primary_metric",
                    "n_splits",
                    "fold_metrics",
                ]:
                    other_metrics[k] = v

            # Display relevant metrics based on problem type
            if reg_metrics:
                st.markdown("#### 📈 Regression Metrics")
                cols = st.columns(min(4, len(reg_metrics)))
                for idx, (metric_name, metric_value) in enumerate(reg_metrics.items()):
                    with cols[idx % len(cols)]:
                        if isinstance(metric_value, (int, float)):
                            # Check if there's a corresponding std metric
                            std_key = metric_name.replace("mean_", "std_")
                            if is_cv and std_key in best_model_metrics:
                                std_value = best_model_metrics[std_key]
                                display_name = (
                                    metric_name.replace("mean_", "")
                                    .upper()
                                    .replace("_", " ")
                                )
                                st.metric(
                                    display_name,
                                    f"{metric_value:.4f}",
                                    delta=f"±{std_value:.4f}",
                                )
                            else:
                                st.metric(
                                    metric_name.upper()
                                    .replace("_", " ")
                                    .replace("MEAN ", ""),
                                    f"{metric_value:.4f}",
                                )
                        else:
                            st.metric(
                                metric_name.replace("_", " ")
                                .title()
                                .replace("Mean ", ""),
                                str(metric_value),
                            )

            if class_metrics:
                st.markdown("#### 🎯 Classification Metrics")
                cols = st.columns(min(4, len(class_metrics)))
                for idx, (metric_name, metric_value) in enumerate(
                    class_metrics.items()
                ):
                    with cols[idx % len(cols)]:
                        if isinstance(metric_value, (int, float)):
                            # Check if there's a corresponding std metric
                            std_key = metric_name.replace("mean_", "std_")
                            if is_cv and std_key in best_model_metrics:
                                std_value = best_model_metrics[std_key]
                                display_name = (
                                    metric_name.replace("mean_", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                st.metric(
                                    display_name,
                                    f"{metric_value:.4f}",
                                    delta=f"±{std_value:.4f}",
                                )
                            else:
                                st.metric(
                                    metric_name.replace("_", " ")
                                    .title()
                                    .replace("Mean ", ""),
                                    f"{metric_value:.4f}",
                                )
                        else:
                            st.metric(
                                metric_name.replace("_", " ")
                                .title()
                                .replace("Mean ", ""),
                                str(metric_value),
                            )

            if other_metrics:
                st.markdown("#### 📊 Additional Metrics")
                cols = st.columns(min(4, len(other_metrics)))
                for idx, (metric_name, metric_value) in enumerate(
                    other_metrics.items()
                ):
                    with cols[idx % len(cols)]:
                        if isinstance(metric_value, (int, float)):
                            # Check if there's a corresponding std metric
                            std_key = metric_name.replace("mean_", "std_")
                            if is_cv and std_key in best_model_metrics:
                                std_value = best_model_metrics[std_key]
                                display_name = (
                                    metric_name.replace("mean_", "")
                                    .replace("_", " ")
                                    .title()
                                )
                                st.metric(
                                    display_name,
                                    f"{metric_value:.4f}",
                                    delta=f"±{std_value:.4f}",
                                )
                            else:
                                st.metric(
                                    metric_name.replace("_", " ")
                                    .title()
                                    .replace("Mean ", ""),
                                    f"{metric_value:.4f}",
                                )
                        else:
                            st.metric(
                                metric_name.replace("_", " ")
                                .title()
                                .replace("Mean ", ""),
                                str(metric_value),
                            )

            # Show all metrics in a table as well
            st.markdown("---")
            st.markdown("#### 📋 All Metrics")
            # Filter out internal keys for cleaner display
            display_metrics = {
                k: v
                for k, v in best_model_metrics.items()
                if k
                not in ["fold_metrics", "mean_primary_metric", "std_primary_metric"]
            }
            metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": k.replace("mean_", "")
                        .replace("std_", "STD ")
                        .replace("_", " ")
                        .title(),
                        "Value": f"{v:.6f}" if isinstance(v, (int, float)) else str(v),
                    }
                    for k, v in display_metrics.items()
                ]
            )
            st.dataframe(metrics_df, width="stretch", hide_index=True)
        else:
            st.warning(
                "⚠️ No detailed metrics available. Metrics may not have been computed during training."
            )
            st.info(
                "💡 Try retraining with cross-validation enabled for detailed metrics."
            )

            # Debug information (collapsible)
            with st.expander("🔍 Debug: View Results Structure"):
                st.write("**Best model name:**", best_model_name)
                st.write("**Best model result found:**", best_model_result is not None)
                if best_model_result:
                    st.write("**Result keys:**", list(best_model_result.keys()))
                    if "cross_validation" in best_model_result:
                        st.json(best_model_result["cross_validation"])
                    elif "val_metrics" in best_model_result:
                        st.json(best_model_result["val_metrics"])
                    elif "train_metrics" in best_model_result:
                        st.json(best_model_result["train_metrics"])

    with tab3:
        st.markdown("### 📄 Export Results")

        # Ensure datetime is available in this scope
        from datetime import datetime as dt

        # PDF Report Generation
        st.markdown("#### 📑 Comprehensive Report")
        col_pdf1, col_pdf2 = st.columns([2, 1])

        with col_pdf1:
            st.info(
                "📄 Generate a professional PDF report with all results, charts, and metrics."
            )

        with col_pdf2:
            if st.button("📥 Generate PDF Report", type="primary", width="stretch"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_buffer = generate_pdf_report(results)
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"automl_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )
                        st.success("✅ PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

        st.markdown("---")
        st.markdown("#### 💾 Data Exports")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export results as JSON
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"automl_results_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch",
            )

        with col2:
            # Export rankings as CSV
            if rankings:
                csv = pd.DataFrame(rankings).to_csv(index=False)
                st.download_button(
                    label="📥 Download Rankings (CSV)",
                    data=csv,
                    file_name=f"model_rankings_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with col3:
            # Download trained model
            best_model_bytes = results.get("best_model_bytes")
            if best_model_bytes:
                try:
                    st.download_button(
                        label="🤖 Download Trained Model",
                        data=best_model_bytes,
                        file_name=f"best_model_{results.get('best_model', 'model')}_{dt.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        width="stretch",
                        help="Download the trained model as a pickle file for later use",
                    )
                except Exception as e:
                    st.error(f"Error creating model download: {str(e)}")
            else:
                st.info("💡 Model saved to MLflow")

        st.markdown("---")

        # Model info
        st.markdown("### 🤖 Best Model Information")

        # Show basic model info from results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Model Details:**")
            st.info(f"""
            **Model:** {results.get('best_model', 'N/A')}  
            **{metric_label}:** {best_model_accuracy:.4f}  
            **Problem Type:** {results.get('problem_type', 'N/A')}  
            **Features:** {results.get('n_features', 'N/A')}  
            **Training Samples:** {results.get('n_samples_train', 'N/A')}
            """)

        with col2:
            st.markdown("**MLflow Tracking:**")
            mlflow_run_id = results.get("mlflow_run_id")
            if mlflow_run_id:
                st.success(f"✅ Model logged to MLflow")
                st.code(f"Run ID: {mlflow_run_id[:8]}...", language="text")
                st.caption("View full details in MLflow UI: `mlflow ui`")
            else:
                st.warning("MLflow tracking not available")

            st.markdown("**Download:**")
            if results.get("best_model_bytes"):
                st.success("✅ Model available for download above")
            else:
                st.info("💡 Load model from MLflow artifacts")

        st.markdown("---")

        # Preprocessing info
        if (
            "preprocessing_config" in st.session_state
            and st.session_state.preprocessing_config
        ):
            st.markdown("### ⚙️ Preprocessing Configuration Used")
            config_df = pd.DataFrame(
                [
                    {"Setting": k.replace("_", " ").title(), "Value": str(v)}
                    for k, v in st.session_state.preprocessing_config.items()
                ]
            )
            st.dataframe(config_df, width="stretch", hide_index=True)


def quick_start_page():
    """Quick start guide and examples."""
    st.title("🚀 Quick Start Guide")

    st.markdown("""
    Welcome to the **AutoML System v1.0.0**! This guide will help you get started with both traditional ML and deep learning models.
    
    ### ✨ Latest Updates (v1.0.0):
    - 🚀 **GPU Acceleration**: Automatic GPU support for XGBoost, LightGBM, CatBoost, and PyTorch models
    - 🧠 **New DL Models**: TabularResNet, TabularAttention, Wide & Deep for tabular data
    - 🖼️ **Enhanced Vision**: VGG16/19, DenseNet121/169 added to CNN architectures
    - 🤖 **Smart Recommendations**: AI suggests ML vs DL based on your dataset
    - ⚡ **27+ Models**: Expanded from 6 to 27 total models available
    """)

    # Feature Overview
    st.markdown("## 🎯 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Tabular Data
        - **Traditional ML (18 models)**: 
          - XGBoost*, LightGBM*, CatBoost* (GPU)
          - Random Forest, Gradient Boosting
          - SVM, KNN, Naive Bayes
          - Ridge, Lasso, ElasticNet
        - **Deep Learning (4 architectures)**:
          - MLP (Multi-Layer Perceptron)
          - TabularResNet (Skip connections)
          - TabularAttention (Transformer)
          - Wide & Deep (Hybrid)
        - Auto preprocessing & feature engineering
        - HPO with Optuna
        
        *GPU accelerated
        """)

    with col2:
        st.markdown("""
        ### 🖼️ Image Data
        - **Custom CNNs**: SimpleCNN, MediumCNN
        - **Transfer Learning (9 models)**:
          - ResNet18, ResNet50
          - VGG16, VGG19
          - DenseNet121, DenseNet169
          - EfficientNet-B0
        - Data augmentation
        - GPU acceleration
        - Fine-tuning support
        - Pretrained weights
        """)

    with col3:
        st.markdown("""
        ### 📝 Text Data
        - **RNNs**: LSTM, GRU
        - **TextCNN**: Fast classification
        - **Attention**: Best accuracy
        - Automatic tokenization
        - Embedding support
        - Multi-class support
        - GPU acceleration
        """)

    st.markdown("---")

    # GPU Support Info
    st.markdown("## ⚡ GPU Acceleration")

    st.info("""
    **Automatic GPU Support** - The system automatically detects and uses GPU when available:
    
    - **Traditional ML**: XGBoost, LightGBM, and CatBoost use GPU automatically (2-10x speedup)
    - **Deep Learning**: All PyTorch models use GPU by default (5-20x speedup)
    - **Fallback**: Seamlessly falls back to CPU if GPU is not available
    - **Status**: GPU status is shown in the Hardware Status section during training
    
    💡 **Requirements**: CUDA-capable GPU + CUDA toolkit + GPU-enabled PyTorch
    """)

    st.markdown("---")

    # Workflow
    st.markdown("## 📋 Workflow")

    st.markdown("""
    ### Step 1: Upload Data
    Navigate to **📁 Data Upload** and:
    - Select your data type (Tabular, Image, or Text)
    - Upload your dataset
    - Preview the data
    
    ### Step 2: Explore (Optional)
    Go to **📊 EDA Dashboard** to:
    - View statistical summaries
    - Analyze distributions
    - Check correlations
    - Identify missing values
    
    ### Step 3: Train Models
    Head to **🎯 Train Models** to:
    - Choose model family (Traditional ML or Deep Learning)
    - Configure hyperparameters
    - Enable advanced preprocessing
    - Start training
    
    ### Step 4: Review Results
    Check **📈 Results & Comparison** for:
    - Model performance metrics
    - Training history (DL models)
    - Model comparison
    - Export trained models
    """)

    st.markdown("---")

    # Examples
    st.markdown("## 💡 Examples")

    tab1, tab2, tab3 = st.tabs(["📊 Tabular", "🖼️ Image", "📝 Text"])

    with tab1:
        st.markdown("### Tabular Data Example")
        st.code(
            """
# Traditional ML (AutoML)
1. Upload CSV file (e.g., Titanic dataset)
2. Select target column (e.g., 'Survived')
3. Choose 'Traditional ML (AutoML)'
4. Click 'Start Training'
5. View model rankings

# Deep Learning (MLP)
1. Upload CSV file
2. Select target column
3. Choose 'Deep Learning (PyTorch)'
4. Configure MLP:
   - Hidden dims: 128,64,32
   - Activation: relu
   - Dropout: 0.3
   - Epochs: 50
5. Click 'Start Training'
6. View training curves
        """,
            language="python",
        )

        st.info(
            "💡 **Tip**: Use MLP for complex tabular patterns, Traditional ML for interpretability"
        )

    with tab2:
        st.markdown("### Image Classification Example")
        st.code(
            """
# CNN Transfer Learning
1. Prepare ZIP file with structure:
   dataset.zip
   ├── class1/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── class2/
       ├── img3.jpg
       └── img4.jpg

2. Select 'Image Data' in Data Upload
3. Upload ZIP file OR specify folder path
   - ZIP: Upload compressed folder structure
   - Folder: Enter path like 'D:/datasets/my_images'
4. In Train Models:
   - Choose 'Deep Learning (PyTorch)'
   - Architecture: 'resnet18' (recommended)
   - Image size: 224
   - Enable augmentation
   - Epochs: 30
5. Click 'Start Training'
6. View accuracy curves
        """,
            language="python",
        )

        st.info(
            "💡 **Tip**: Use ResNet18 for small datasets, ResNet50/EfficientNet for larger datasets"
        )

    with tab3:
        st.markdown("### Text Classification Example")
        st.code(
            """
# NLP Sentiment Analysis
1. Prepare CSV with columns:
   - 'text': Your text samples
   - 'label': Categories (e.g., positive/negative)

2. Upload CSV in Text Data mode
3. Select text and label columns
4. In Train Models:
   - Choose 'Deep Learning (PyTorch)'
   - Architecture: 'lstm' (recommended)
   - Embedding dim: 100
   - Hidden dim: 128
   - Max length: 128
   - Bidirectional: Yes
   - Epochs: 20
5. Click 'Start Training'
6. View predictions and accuracy

# Architecture Guide:
- LSTM: Best for sequential patterns
- GRU: Faster than LSTM, similar performance
- CNN: Fast, good for keyword-based classification
- Attention: Best accuracy, slower training
        """,
            language="python",
        )

        st.info("💡 **Tip**: Start with LSTM, try Attention for better accuracy")

    st.markdown("---")

    # Tips & Best Practices
    st.markdown("## 🎓 Tips & Best Practices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### General Tips
        - **Start simple**: Use default settings first
        - **Check data quality**: Handle missing values and outliers
        - **Use cross-validation**: More robust evaluation
        - **Enable MLflow**: Track all experiments
        """)

        st.markdown("""
        ### Traditional ML
        - **Feature engineering**: Often improves performance
        - **Feature selection**: Reduces overfitting
        - **HPO**: Let Optuna find best hyperparameters
        - **Ensemble**: Combine multiple models
        """)

    with col2:
        st.markdown("""
        ### Deep Learning
        - **GPU**: Enable for 10-50x speedup
        - **Batch size**: Larger = faster, but needs more memory
        - **Learning rate**: Start with 0.001
        - **Early stopping**: Prevents overfitting
        - **Data augmentation**: Essential for images
        """)

        st.markdown("""
        ### Troubleshooting
        - **Low accuracy**: Try more epochs, different architecture
        - **Overfitting**: Increase dropout, reduce model size
        - **Slow training**: Reduce batch size, use simpler model
        - **Out of memory**: Reduce batch size or image size
        """)

    st.markdown("---")

    # Code Examples
    st.markdown("## 💻 Code Examples (Programmatic Use)")

    st.markdown("### Traditional ML")
    st.code(
        """
from automl.pipeline.automl import AutoML
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Create AutoML instance
automl = AutoML(
    problem_type='classification',
    optimize_hyperparameters=True,
    n_trials=20,
    enable_mlflow=True
)

# Train
results = automl.fit(df, target_column='target')

# Get best model
best_model = automl.best_model
predictions = best_model.predict(test_data)
    """,
        language="python",
    )

    st.markdown("### Deep Learning - MLP")
    st.code(
        """
from automl.models.deep_learning.mlp_models import MLPClassifier

model = MLPClassifier(
    hidden_dims=[128, 64, 32],
    activation='relu',
    dropout=0.3,
    batch_size=32,
    max_epochs=50,
    learning_rate=0.001,
    use_gpu=True
)

model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
    """,
        language="python",
    )

    st.markdown("### Deep Learning - CNN")
    st.code(
        """
from automl.models.deep_learning.computer_vision import CNNClassifier

model = CNNClassifier(
    architecture='resnet18',
    num_classes=10,
    image_size=224,
    batch_size=32,
    max_epochs=30,
    use_augmentation=True,
    use_gpu=True
)

model.fit(image_paths, labels)
predictions = model.predict(test_image_paths)
    """,
        language="python",
    )

    st.markdown("### Deep Learning - NLP")
    st.code(
        """
from automl.models.deep_learning.nlp import TextClassifier

model = TextClassifier(
    architecture='lstm',
    embedding_dim=100,
    hidden_dim=128,
    max_length=128,
    batch_size=32,
    max_epochs=20,
    use_gpu=True
)

model.fit(train_texts, train_labels)
predictions = model.predict(test_texts)
probabilities = model.predict_proba(test_texts)
    """,
        language="python",
    )

    st.markdown("---")

    # Resources
    st.markdown("## 📚 Resources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Documentation
        - [Quick Start Guide](QUICK_START_GUIDE.md)
        - [Web App README](WEB_APP_README.md)
        - [Month 7: MLP](MONTH_7_OPTIMIZATIONS.md)
        """)

    with col2:
        st.markdown("""
        ### Deep Learning Docs
        - [Month 8: Computer Vision](MONTH_8_COMPUTER_VISION.md)
        - [Month 9: NLP](MONTH_9_NLP.md)
        - [Month 9 Report](MONTH_9_COMPLETION_REPORT.md)
        """)

    with col3:
        st.markdown("""
        ### Examples
        - `examples/automl_example.py`
        - `examples/nlp_example.py`
        - `examples/computer_vision_example.py`
        """)

    st.success("🎉 Ready to start? Go to **📁 Data Upload** to begin!")


if __name__ == "__main__":
    main()
