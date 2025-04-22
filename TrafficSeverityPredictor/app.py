"""
Traffic Severity Prediction App - Main Application
"""

import streamlit as st
import pandas as pd
from utils.data_utils import get_sample_data

# Set page configuration
st.set_page_config(
    page_title="Traffic Severity Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data if not exists
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = get_sample_data(500)

# Main page content
st.title("🚦 Traffic Severity Prediction System")
st.subheader("Predict the severity of traffic incidents based on environmental factors")

# Overview section
st.markdown("""
## Overview

This application allows you to predict the severity of traffic incidents based on environmental and situational factors.
Using machine learning, it analyzes patterns in traffic data to provide insights on potential severity outcomes.

### Features

- **Predict** incident severity based on various factors
- **Visualize** prediction results and historical data
- **Explore** traffic incident patterns and correlations
- **Analyze** model performance and feature importance

### Getting Started

Use the sidebar to navigate to different sections of the application:

1. **Prediction**: Input parameters and get severity predictions
2. **Dashboard**: View prediction history and statistics
3. **Data Exploration**: Analyze traffic incident patterns
4. **Model Performance**: Evaluate model metrics and feature importance

You can also train and integrate your own custom prediction models by following the guide in `guides/adding_custom_model.md`.
""")

# Display a sample of the data
st.subheader("Sample Traffic Incident Data")
st.dataframe(st.session_state.sample_data.head(10))

# Quick stats
st.subheader("Quick Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Sample Incidents", 
        value=len(st.session_state.sample_data)
    )

with col2:
    severity_counts = st.session_state.sample_data["Severity"].value_counts()
    high_severity_pct = round(severity_counts.get("High", 0) / len(st.session_state.sample_data) * 100, 1)
    st.metric(
        label="High Severity Incidents", 
        value=f"{high_severity_pct}%"
    )

with col3:
    avg_clearance = round(st.session_state.sample_data["Clearance_Time"].mean(), 1)
    st.metric(
        label="Avg. Clearance Time", 
        value=f"{avg_clearance} minutes"
    )

# Footer
st.markdown("---")
st.markdown("© 2025 Traffic Severity Prediction System")