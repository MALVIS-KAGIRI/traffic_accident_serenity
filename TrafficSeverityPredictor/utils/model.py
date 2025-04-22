"""
Model utilities for the Traffic Severity Prediction App
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from utils.database import save_prediction

def initialize_model():
    """
    Initialize and return a pre-trained decision tree model.
    
    In a real application, this would load a saved model that was properly trained
    on historical data. For demonstration purposes, we're creating a new model.
    """
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Train on synthetic data
    X, y = _generate_training_data(1000)
    model.fit(X, y)
    
    return model

def predict_severity(model, input_data):
    """
    Make a prediction using the provided model and input data.
    
    Args:
        model: The trained decision tree model
        input_data: Dictionary containing feature values
        
    Returns:
        tuple: (predicted_severity, confidence_percentage, explanation)
    """
    # Process input features for the model
    processed_input = _process_input_features(input_data)
    
    # Make the prediction
    severity_label = model.predict([processed_input])[0]
    
    # Get prediction probabilities
    proba = model.predict_proba([processed_input])[0]
    confidence = max(proba) * 100  # Convert to percentage
    
    # Generate an explanation for the prediction
    explanation = _generate_explanation(model, processed_input, input_data)
    
    # Save the prediction to the database
    try:
        prediction_data = input_data.copy()
        prediction_data['Prediction'] = severity_label
        prediction_data['Confidence'] = confidence
        save_prediction(prediction_data)
    except Exception as e:
        print(f"Error saving prediction to database: {e}")
    
    return severity_label, confidence, explanation

def get_model_metrics():
    """
    Return metrics about the model's performance.
    
    In a real application, these would be calculated from validation data.
    For demonstration, we're generating synthetic metrics.
    """
    # Generate test data
    X_test, y_test = _generate_training_data(200)
    
    # Initialize model
    model = initialize_model()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    
    # Calculate feature importance
    feature_importance = list(zip(
        ['Day of Week', 'Hour', 'Weather', 'Road Type', 'Junction Type', 'Traffic Volume', 'Speed Limit'],
        model.feature_importances_
    ))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'confusion_matrix': {
            'true_positive': int(sum((y_pred == 'High') & (y_test == 'High'))),
            'false_positive': int(sum((y_pred == 'High') & (y_test != 'High'))),
            'true_negative': int(sum((y_pred != 'High') & (y_test != 'High'))),
            'false_negative': int(sum((y_pred != 'High') & (y_test == 'High')))
        }
    }

def _process_input_features(input_data):
    """
    Process and encode input features for the model.
    
    Args:
        input_data: Dictionary of raw input features
        
    Returns:
        numpy.ndarray: Processed features ready for the model
    """
    # Extract features in the correct order
    day_of_week = input_data['Day_of_Week']
    hour = input_data['Hour']
    weather = input_data['Weather_Condition']
    road_type = input_data['Road_Type']
    junction_type = input_data['Junction_Type']
    traffic_volume = input_data['Traffic_Volume']
    speed_limit = input_data['Speed_Limit']
    
    # One-hot encode categorical features (simplified for demo)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_idx = days.index(day_of_week)
    
    weather_types = ['Clear', 'Rain', 'Snow', 'Fog', 'Windy']
    weather_idx = weather_types.index(weather)
    
    road_types = ['Highway', 'Urban', 'Residential', 'Rural']
    road_idx = road_types.index(road_type)
    
    junction_types = ['Intersection', 'Roundabout', 'T-Junction', 'None']
    junction_idx = junction_types.index(junction_type)
    
    # Normalize numerical features
    traffic_norm = (traffic_volume - 50) / (5000 - 50)  # Normalize to [0, 1]
    speed_norm = (speed_limit - 25) / (75 - 25)  # Normalize to [0, 1]
    hour_norm = hour / 23  # Normalize to [0, 1]
    
    # For a real decision tree, we'd use one-hot encoding, but for simplicity
    # we'll just use the indices as features
    return np.array([
        day_idx / 6,  # Normalize to [0, 1]
        hour_norm,
        weather_idx / 4,  # Normalize to [0, 1]
        road_idx / 3,  # Normalize to [0, 1]
        junction_idx / 3,  # Normalize to [0, 1]
        traffic_norm,
        speed_norm
    ])

def _generate_explanation(model, processed_input, raw_input):
    """
    Generate an explanation for the model's prediction.
    
    Args:
        model: The decision tree model
        processed_input: Processed input features
        raw_input: Raw input features
        
    Returns:
        list: List of (factor, importance) tuples explaining the prediction
    """
    # For a real application, you'd use SHAP values or a similar explainability method
    # For demonstration, we'll create a simplified explanation based on feature importance
    
    # Get feature importance from the model
    importance = model.feature_importances_
    
    # Map feature indices to names and values
    features = [
        ('Day_of_Week', raw_input['Day_of_Week']),
        ('Hour', raw_input['Hour']),
        ('Weather_Condition', raw_input['Weather_Condition']),
        ('Road_Type', raw_input['Road_Type']),
        ('Junction_Type', raw_input['Junction_Type']),
        ('Traffic_Volume', raw_input['Traffic_Volume']),
        ('Speed_Limit', raw_input['Speed_Limit'])
    ]
    
    # Weight the features by their importance
    weighted_features = [(f"{name}: {value}", float(imp)) for (name, value), imp in zip(features, importance)]
    
    # Sort by importance (highest first) and return top 3
    weighted_features.sort(key=lambda x: x[1], reverse=True)
    return [(factor, f"{importance:.2f}") for factor, importance in weighted_features[:3]]

def _generate_training_data(n_samples):
    """
    Generate synthetic training data for demonstration purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        tuple: (X, y) where X is features and y is target labels
    """
    # Generate random features
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_indices = np.random.randint(0, 7, n_samples)
    
    hours = np.random.randint(0, 24, n_samples)
    
    weather_types = ['Clear', 'Rain', 'Snow', 'Fog', 'Windy']
    weather_indices = np.random.randint(0, 5, n_samples)
    
    road_types = ['Highway', 'Urban', 'Residential', 'Rural']
    road_indices = np.random.randint(0, 4, n_samples)
    
    junction_types = ['Intersection', 'Roundabout', 'T-Junction', 'None']
    junction_indices = np.random.randint(0, 4, n_samples)
    
    traffic_volumes = np.random.randint(50, 5000, n_samples)
    speed_limits = np.random.choice([25, 35, 45, 55, 65, 75], n_samples)
    
    # Normalize numerical features
    traffic_norm = (traffic_volumes - 50) / (5000 - 50)
    speed_norm = (speed_limits - 25) / (75 - 25)
    hour_norm = hours / 23
    
    # Create feature matrix
    X = np.column_stack([
        day_indices / 6,
        hour_norm,
        weather_indices / 4,
        road_indices / 3,
        junction_indices / 3,
        traffic_norm,
        speed_norm
    ])
    
    # Create synthetic rules for severity prediction
    # These are simplified rules just for demonstration
    severity = []
    for i in range(n_samples):
        score = 0
        
        # Weather affects severity
        if weather_indices[i] in [1, 2, 3]:  # Rain, Snow, Fog
            score += 0.3
        
        # High traffic volume increases severity
        if traffic_volumes[i] > 3000:
            score += 0.3
        elif traffic_volumes[i] > 1500:
            score += 0.15
            
        # Junction type affects severity
        if junction_indices[i] in [0, 1]:  # Intersection, Roundabout
            score += 0.2
            
        # High speed limit increases severity
        if speed_limits[i] >= 65:
            score += 0.3
        elif speed_limits[i] >= 45:
            score += 0.15
            
        # Night hours (7pm-5am) increase severity
        if hours[i] >= 19 or hours[i] <= 5:
            score += 0.2
            
        # Weekend days might increase severity
        if day_indices[i] >= 5:  # Saturday, Sunday
            score += 0.1
            
        # Assign severity based on score
        if score >= 0.6:
            severity.append('High')
        elif score >= 0.3:
            severity.append('Medium')
        else:
            severity.append('Low')
    
    return X, np.array(severity)