"""
Traffic Severity Prediction App - Dashboard Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.database import get_prediction_history

# Set page title
st.set_page_config(
    page_title="Dashboard - Traffic Severity Prediction",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("📊 Prediction Dashboard")
st.markdown("View and analyze your prediction history")

# Get prediction history from database
prediction_history = get_prediction_history()

# Display dashboard or info message
if len(prediction_history) == 0:
    st.info("No predictions made yet. Go to the Prediction page to make predictions.")
else:
    # Dashboard metrics
    st.subheader("Summary Metrics")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(prediction_history))
    
    with col2:
        severity_counts = prediction_history['Prediction'].value_counts()
        high_percent = round(severity_counts.get('High', 0) / len(prediction_history) * 100, 1)
        st.metric("High Severity Predictions", f"{high_percent}%")
    
    with col3:
        avg_confidence = round(prediction_history['Confidence'].mean(), 1)
        st.metric("Average Confidence", f"{avg_confidence}%")
    
    with col4:
        recent_trend = 0
        if len(prediction_history) >= 5:
            recent = prediction_history.iloc[-5:]['Prediction'].value_counts(normalize=True).get('High', 0) * 100
            overall = prediction_history['Prediction'].value_counts(normalize=True).get('High', 0) * 100
            recent_trend = round(recent - overall, 1)
        
        st.metric("Recent High Severity Trend", f"{recent_trend:+}%", delta_color="inverse")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution pie chart
        st.subheader("Severity Distribution")
        fig = px.pie(
            prediction_history, 
            names='Prediction',
            color='Prediction',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Road type vs severity
        st.subheader("Road Type vs Severity")
        road_severity = pd.crosstab(
            prediction_history['Road_Type'], 
            prediction_history['Prediction'],
            normalize='index'
        ) * 100
        
        fig = px.bar(
            road_severity,
            barmode='stack',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        )
        fig.update_layout(height=400, yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Day of week distribution
        st.subheader("Predictions by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = prediction_history['Day_of_Week'].value_counts().reindex(day_order, fill_value=0)
        
        fig = px.bar(
            x=day_counts.index, 
            y=day_counts.values,
            labels={'x': 'Day of Week', 'y': 'Number of Predictions'},
            color_discrete_sequence=['#2E86C1']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hour distribution heatmap
        st.subheader("Predictions by Hour and Severity")
        hour_severity = pd.crosstab(prediction_history['Hour'], prediction_history['Prediction'])
        
        fig = go.Figure(data=go.Heatmap(
            z=hour_severity.values,
            x=hour_severity.columns,
            y=hour_severity.index,
            colorscale='YlOrRd',
            hoverongaps=False
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Severity",
            yaxis_title="Hour of Day"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.subheader("Recent Predictions")
    st.dataframe(
        prediction_history.sort_values('Timestamp', ascending=False).head(10),
        use_container_width=True,
        column_order=['Timestamp', 'Prediction', 'Confidence', 'Day_of_Week', 'Hour', 
                     'Weather_Condition', 'Road_Type', 'Junction_Type', 'Traffic_Volume', 'Speed_Limit'],
    )
    
    # Additional insights
    st.subheader("Key Insights")
    
    # Calculate some insights from the data
    weather_high = prediction_history.groupby('Weather_Condition')['Prediction'].apply(
        lambda x: (x == 'High').mean() * 100
    ).sort_values(ascending=False)
    
    junction_high = prediction_history.groupby('Junction_Type')['Prediction'].apply(
        lambda x: (x == 'High').mean() * 100
    ).sort_values(ascending=False)
    
    # Display insights
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Weather Conditions with Highest Severity Risk:**")
        for weather, pct in weather_high.items():
            if pct > 0:
                st.markdown(f"- {weather}: {pct:.1f}% high severity risk")
    
    with insights_col2:
        st.markdown("**Junction Types with Highest Severity Risk:**")
        for junction, pct in junction_high.items():
            if pct > 0:
                st.markdown(f"- {junction}: {pct:.1f}% high severity risk")