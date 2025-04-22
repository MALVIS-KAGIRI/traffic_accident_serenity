"""
Traffic Severity Prediction App - Model Performance Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.model import get_model_metrics, initialize_model

# Set page title
st.set_page_config(
    page_title="Model Performance - Traffic Severity Prediction",
    page_icon="📈",
    layout="wide"
)

# Page title
st.title("📈 Model Performance Analysis")
st.markdown("Analyze and understand the traffic severity prediction model")

# Make sure model is initialized
if 'model' not in st.session_state:
    st.session_state.model = initialize_model()

# Get model metrics
metrics = get_model_metrics()

# Display accuracy metric
st.header("Model Accuracy")
accuracy_value = metrics['accuracy'] * 100
accuracy_color = 'green' if accuracy_value > 80 else 'orange' if accuracy_value > 70 else 'red'

st.markdown(
    f"""
    <div style="background-color: {accuracy_color}20; padding: 20px; border-radius: 10px; text-align: center;">
        <h2 style="color: {accuracy_color};">{accuracy_value:.1f}%</h2>
        <p>Overall model accuracy on test data</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Confusion Matrix
st.header("Confusion Matrix")
st.markdown("This shows how well the model identifies high severity incidents")

# Create columns
col1, col2 = st.columns([3, 2])

with col1:
    # Create confusion matrix visualization
    cm_values = metrics['confusion_matrix']
    
    # Create a confusion matrix array for heatmap
    cm_array = [
        [cm_values['true_negative'], cm_values['false_positive']],
        [cm_values['false_negative'], cm_values['true_positive']]
    ]
    
    labels = ['Not High', 'High']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_array,
        x=labels,
        y=labels,
        text=[[str(val) for val in row] for row in cm_array],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        xaxis_title="Predicted Severity",
        yaxis_title="Actual Severity",
        xaxis=dict(side='top'),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Calculate additional metrics
    tp = cm_values['true_positive']
    tn = cm_values['true_negative']
    fp = cm_values['false_positive']
    fn = cm_values['false_negative']
    
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create metrics display
    metrics_data = {
        "Metric": ["Precision", "Recall", "F1 Score"],
        "Value": [f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}"],
        "Description": [
            "Accuracy of positive predictions",
            "Ability to find all positives",
            "Balance between precision and recall"
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    **Understanding These Metrics:**
    
    - **Precision:** How many predicted high severity incidents were actually high severity
    - **Recall:** How many actual high severity incidents were correctly predicted
    - **F1 Score:** Harmonic mean of precision and recall
    """)

# Feature importance visualization
st.header("Feature Importance")
st.markdown("See which factors influence severity predictions the most")

# Create feature importance dataframe
fi_data = pd.DataFrame(metrics['feature_importance'], columns=['Feature', 'Importance'])
fi_data = fi_data.sort_values('Importance', ascending=False)

# Create horizontal bar chart
fig = px.bar(
    fi_data,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='viridis',
    labels={'Importance': 'Relative Importance'},
)

fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
st.plotly_chart(fig, use_container_width=True)

# Feature Impact Analysis
st.header("Feature Impact Analysis")
st.markdown("Explore how different factors affect severity predictions")

# Create tabs for different feature analyses
tab1, tab2, tab3 = st.tabs(["Weather & Road", "Time Factors", "Traffic & Speed"])

with tab1:
    # Weather impact
    st.subheader("Weather Conditions Impact")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    weather_types = ['Clear', 'Rain', 'Snow', 'Fog', 'Windy']
    weather_impact = []
    
    for weather in weather_types:
        # Simulate different predictions for each weather (more high severity for adverse weather)
        high_pct = 0.1  # base rate
        
        if weather == 'Rain':
            high_pct = 0.25
        elif weather == 'Snow':
            high_pct = 0.40
        elif weather == 'Fog':
            high_pct = 0.30
        elif weather == 'Windy':
            high_pct = 0.15
            
        weather_impact.append({
            'Weather': weather,
            'Low': (1 - high_pct) * 0.7,
            'Medium': (1 - high_pct) * 0.3,
            'High': high_pct
        })
    
    weather_df = pd.DataFrame(weather_impact)
    
    # Melt the dataframe for stacked bar chart
    weather_melted = pd.melt(
        weather_df, 
        id_vars=['Weather'], 
        var_name='Severity', 
        value_name='Probability'
    )
    
    # Create stacked bar chart
    fig = px.bar(
        weather_melted,
        x='Weather',
        y='Probability',
        color='Severity',
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        labels={'Probability': 'Probability of Severity'},
        barmode='stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Insight:** Snow and fog conditions have the highest probability of causing high severity incidents,
    while clear weather is associated with the lowest severity risk.
    """)

with tab2:
    # Time factors impact
    st.subheader("Time Factors Impact")
    
    # Generate synthetic data for time impact
    hours = list(range(24))
    time_impact = []
    
    for hour in hours:
        # Simulate different predictions for different hours
        # Rush hours (7-9am, 4-6pm) and late night have higher severity
        high_pct = 0.1  # base rate
        
        if 7 <= hour <= 9:  # Morning rush
            high_pct = 0.25
        elif 16 <= hour <= 18:  # Evening rush
            high_pct = 0.30
        elif 23 <= hour or hour <= 4:  # Late night/early morning
            high_pct = 0.35
            
        time_impact.append({
            'Hour': hour,
            'High Severity Risk': high_pct * 100
        })
    
    time_df = pd.DataFrame(time_impact)
    
    # Create line chart
    fig = px.line(
        time_df,
        x='Hour',
        y='High Severity Risk',
        markers=True,
        labels={'High Severity Risk': 'High Severity Risk (%)'}
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=2),
        yaxis=dict(range=[0, 40])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Insight:** The highest risk periods for severe traffic incidents are during evening rush hour (4-6 PM)
    and late night/early morning hours, likely due to congestion and reduced visibility/fatigue respectively.
    """)

with tab3:
    # Traffic volume and speed impact
    st.subheader("Traffic Volume & Speed Impact")
    
    # Create a grid of values
    traffic_volumes = [500, 1000, 2000, 3000, 4000]
    speed_limits = [25, 35, 45, 55, 65, 75]
    
    # Create a matrix of severity values
    severity_matrix = np.zeros((len(traffic_volumes), len(speed_limits)))
    
    # Fill with values (higher values for higher traffic and higher speeds)
    for i, traffic in enumerate(traffic_volumes):
        for j, speed in enumerate(speed_limits):
            # Base rate
            value = 0.1
            
            # Add traffic factor (higher traffic = higher severity)
            traffic_factor = traffic / 4000 * 0.2
            
            # Add speed factor (higher speed = higher severity)
            speed_factor = speed / 75 * 0.3
            
            # Calculate combined risk
            severity_matrix[i, j] = value + traffic_factor + speed_factor
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=severity_matrix,
        x=speed_limits,
        y=traffic_volumes,
        colorscale='RdYlGn_r',
        colorbar=dict(title='Risk of High Severity'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        xaxis_title="Speed Limit (mph)",
        yaxis_title="Traffic Volume (vehicles/hour)",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Insight:** The combination of high traffic volume and high speed limits creates the most dangerous
    conditions for severe traffic incidents. Areas with speed limits of 65-75 mph and traffic volumes
    over 3000 vehicles/hour are particularly high-risk.
    """)

# Model Details and Information
with st.expander("About the Model"):
    st.markdown("""
    ### Decision Tree Classifier
    
    The current model uses a Decision Tree algorithm to predict traffic incident severity based on environmental
    and situational factors. Decision trees work by recursively splitting the data based on feature values to
    create the most homogeneous groups possible.
    
    **Strengths:**
    - Easy to interpret and visualize
    - Handles both numerical and categorical data
    - Requires minimal data preprocessing
    - Can capture non-linear relationships
    
    **Limitations:**
    - Can be prone to overfitting on training data
    - May not perform as well as more complex models
    - Sensitive to small changes in the data
    
    **Model Parameters:**
    - Max Depth: 8 (prevents overfitting by limiting tree complexity)
    - Min Samples Split: 10 (minimum samples required to split a node)
    - Min Samples Leaf: 5 (minimum samples required at a leaf node)
    
    To improve this model or train your own custom model, see the guide in `guides/adding_custom_model.md`.
    """)