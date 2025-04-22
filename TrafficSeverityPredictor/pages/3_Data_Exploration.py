"""
Traffic Severity Prediction App - Data Exploration Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.data_utils import get_sample_data, calculate_statistics
from utils.database import get_traffic_incidents

# Set page title
st.set_page_config(
    page_title="Data Exploration - Traffic Severity Prediction",
    page_icon="🔍",
    layout="wide"
)

# Page title
st.title("🔍 Traffic Data Exploration")
st.markdown("Explore and analyze historical traffic incident data")

# Get data from the database
try:
    data = get_traffic_incidents()
    if len(data) == 0:
        # If no data in database, generate sample data
        data = get_sample_data(500)
except:
    # Fallback to sample data if database error
    data = get_sample_data(500)

# Calculate statistics
stats = calculate_statistics(data)

# Dashboard metrics
st.subheader("Overview Statistics")

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Incidents", stats['total_incidents'])

with col2:
    high_count = stats['severity_distribution'].get('High', 0)
    total = stats['total_incidents']
    high_pct = round(high_count / total * 100, 1) if total > 0 else 0
    st.metric("High Severity Incidents", f"{high_pct}%")

with col3:
    avg_clearance = round(stats['avg_clearance_time'], 1)
    st.metric("Avg. Clearance Time", f"{avg_clearance} minutes")

with col4:
    avg_traffic = round(stats['avg_traffic_volume'], 0)
    st.metric("Avg. Traffic Volume", f"{avg_traffic}")

# Tabs for different analysis views
tab1, tab2, tab3 = st.tabs(["Temporal Patterns", "Environmental Factors", "Advanced Analysis"])

with tab1:
    st.subheader("Temporal Patterns")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Incidents by day of week
        st.subheader("Incidents by Day of Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = pd.Series(stats['incidents_by_day']).reindex(day_order, fill_value=0)
        
        fig = px.bar(
            x=day_counts.index, 
            y=day_counts.values,
            labels={'x': 'Day of Week', 'y': 'Number of Incidents'},
            color_discrete_sequence=['#2E86C1']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Incidents by hour
        st.subheader("Incidents by Hour of Day")
        hour_counts = pd.Series(stats['incidents_by_hour']).sort_index()
        
        fig = px.line(
            x=hour_counts.index, 
            y=hour_counts.values,
            labels={'x': 'Hour of Day', 'y': 'Number of Incidents'},
            markers=True
        )
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap of day and hour
    st.subheader("Incidents by Day and Hour")
    
    # Create a crosstab of day and hour
    day_hour_data = pd.crosstab(data['Day_of_Week'], data['Hour'])
    # Reindex to ensure correct order
    day_hour_data = day_hour_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=day_hour_data.values,
        x=day_hour_data.columns,
        y=day_hour_data.index,
        colorscale='Viridis',
        hoverongaps=False
    ))
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Environmental Factors")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather conditions
        st.subheader("Incidents by Weather Condition")
        weather_counts = pd.Series(stats['incidents_by_weather'])
        
        fig = px.pie(
            values=weather_counts.values,
            names=weather_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather severity relationship
        st.subheader("Weather vs Severity")
        weather_severity = pd.crosstab(
            data['Weather_Condition'], 
            data['Severity'],
            normalize='index'
        ) * 100
        
        fig = px.bar(
            weather_severity,
            barmode='stack',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            labels={'value': 'Percentage (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Road types
        st.subheader("Incidents by Road Type")
        road_counts = pd.Series(stats['incidents_by_road'])
        
        fig = px.bar(
            x=road_counts.index, 
            y=road_counts.values,
            color=road_counts.index,
            labels={'x': 'Road Type', 'y': 'Number of Incidents'},
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Road type vs severity
        st.subheader("Road Type vs Severity")
        road_severity = pd.crosstab(
            data['Road_Type'], 
            data['Severity'],
            normalize='index'
        ) * 100
        
        fig = px.bar(
            road_severity,
            barmode='stack',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            labels={'value': 'Percentage (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Junction vs Speed
    st.subheader("Junction Type vs Speed Limit")
    junction_speed = pd.crosstab(data['Junction_Type'], data['Speed_Limit'])
    
    fig = px.imshow(
        junction_speed, 
        color_continuous_scale='Viridis',
        labels=dict(x="Speed Limit", y="Junction Type", color="Incidents"),
        x=sorted(data['Speed_Limit'].unique()),
        y=data['Junction_Type'].unique()
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Advanced Analysis")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        st.subheader("Feature Correlations")
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        correlation = numeric_data.corr().round(2)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            hoverongaps=False
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Clearance time by severity
        st.subheader("Clearance Time by Severity")
        
        fig = px.box(
            data,
            x='Severity',
            y='Clearance_Time',
            color='Severity',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            labels={'Clearance_Time': 'Clearance Time (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic volume vs severity
        st.subheader("Traffic Volume vs Severity")
        
        fig = px.violin(
            data,
            x='Severity',
            y='Traffic_Volume',
            color='Severity',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            box=True,
            labels={'Traffic_Volume': 'Traffic Volume (vehicles/hour)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Traffic patterns by severity
    st.subheader("Traffic Volume by Hour and Severity")
    
    # Group by hour and severity to get mean traffic volume
    hour_traffic = data.groupby(['Hour', 'Severity'])['Traffic_Volume'].mean().reset_index()
    
    fig = px.line(
        hour_traffic,
        x='Hour',
        y='Traffic_Volume',
        color='Severity',
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        markers=True,
        labels={'Traffic_Volume': 'Average Traffic Volume'}
    )
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=2))
    st.plotly_chart(fig, use_container_width=True)

# Data table with filters
st.subheader("Raw Data Explorer")

# Filters
with st.expander("Filter Data"):
    # Create filter columns
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Filter by day
        days = ['All'] + sorted(data['Day_of_Week'].unique().tolist())
        day_filter = st.selectbox("Day of Week", days)
    
    with filter_col2:
        # Filter by severity
        severities = ['All'] + sorted(data['Severity'].unique().tolist())
        severity_filter = st.selectbox("Severity", severities)
    
    with filter_col3:
        # Filter by weather
        weathers = ['All'] + sorted(data['Weather_Condition'].unique().tolist())
        weather_filter = st.selectbox("Weather Condition", weathers)

# Apply filters
filtered_data = data.copy()

if day_filter != 'All':
    filtered_data = filtered_data[filtered_data['Day_of_Week'] == day_filter]

if severity_filter != 'All':
    filtered_data = filtered_data[filtered_data['Severity'] == severity_filter]

if weather_filter != 'All':
    filtered_data = filtered_data[filtered_data['Weather_Condition'] == weather_filter]

# Show filtered data
st.dataframe(filtered_data, use_container_width=True)