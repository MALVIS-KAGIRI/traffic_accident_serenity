"""
Data utilities for the Traffic Severity Prediction App
"""

import pandas as pd
import numpy as np
import datetime
from utils.database import save_sample_data, get_traffic_incidents

def get_sample_data(n_samples=500):
    """
    Generate sample traffic incident data for exploration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        pandas.DataFrame: DataFrame with sample traffic incident data
    """
    # Try to get data from the database first
    try:
        data = get_traffic_incidents()
        if len(data) > 0:
            return data
    except:
        pass
    
    # Generate synthetic data if no data in database
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weather = ['Clear', 'Rain', 'Snow', 'Fog', 'Windy']
    road_types = ['Highway', 'Urban', 'Residential', 'Rural']
    junction_types = ['Intersection', 'Roundabout', 'T-Junction', 'None']
    severity = ['Low', 'Medium', 'High']
    severity_numeric = {'Low': 1, 'Medium': 2, 'High': 3}
    
    # Create a base date
    base_date = datetime.datetime(2023, 1, 1)
    
    data = {
        'Timestamp': [base_date + datetime.timedelta(hours=np.random.randint(0, 8760)) for _ in range(n_samples)],
        'Day_of_Week': np.random.choice(days, n_samples),
        'Hour': np.random.randint(0, 24, n_samples),
        'Weather_Condition': np.random.choice(weather, n_samples),
        'Road_Type': np.random.choice(road_types, n_samples),
        'Junction_Type': np.random.choice(junction_types, n_samples),
        'Traffic_Volume': np.random.randint(50, 5000, n_samples),
        'Speed_Limit': np.random.choice([25, 35, 45, 55, 65, 75], n_samples),
        'Severity': np.random.choice(severity, n_samples, p=[0.6, 0.3, 0.1])  # Adjust probability distribution
    }
    
    # Add derived features
    df = pd.DataFrame(data)
    df['Severity_Numeric'] = df['Severity'].map(severity_numeric)
    
    # Add clearance time (correlate with severity)
    base_clearance = {1: 30, 2: 60, 3: 120}  # base minutes by severity
    df['Clearance_Time'] = df['Severity_Numeric'].apply(
        lambda x: base_clearance[x] + np.random.randint(-10, 30)
    )
    
    # Save to database
    try:
        save_sample_data(df)
    except Exception as e:
        print(f"Error saving sample data to database: {e}")
    
    return df

def calculate_statistics(data):
    """
    Calculate key statistics from traffic incident data.
    
    Args:
        data: DataFrame with traffic incident data
        
    Returns:
        dict: Dictionary of calculated statistics
    """
    if data is None or len(data) == 0:
        return {
            'total_incidents': 0,
            'severity_distribution': {},
            'avg_clearance_time': 0,
            'incidents_by_day': {},
            'incidents_by_hour': {},
            'incidents_by_weather': {},
            'incidents_by_road': {},
            'avg_traffic_volume': 0
        }
    
    # Calculate basic statistics
    total = len(data)
    severity_counts = data['Severity'].value_counts().to_dict()
    avg_clearance = data['Clearance_Time'].mean()
    
    # Incidents by day, hour, etc.
    by_day = data['Day_of_Week'].value_counts().to_dict()
    by_hour = data['Hour'].value_counts().to_dict()
    by_weather = data['Weather_Condition'].value_counts().to_dict()
    by_road = data['Road_Type'].value_counts().to_dict()
    
    avg_traffic = data['Traffic_Volume'].mean()
    
    return {
        'total_incidents': total,
        'severity_distribution': severity_counts,
        'avg_clearance_time': avg_clearance,
        'incidents_by_day': by_day,
        'incidents_by_hour': by_hour,
        'incidents_by_weather': by_weather,
        'incidents_by_road': by_road,
        'avg_traffic_volume': avg_traffic
    }