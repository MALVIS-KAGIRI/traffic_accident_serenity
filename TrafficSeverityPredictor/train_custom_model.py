"""
Example script to train and save a custom model for the Traffic Severity Prediction app.

This script demonstrates how to:
1. Generate or load training data
2. Train a machine learning model (Random Forest in this example)
3. Save the model to be used in the application
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from utils.data_utils import get_sample_data

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Generating sample training data...")
# In a real scenario, you would load your data from a CSV or database
# data = pd.read_csv('your_traffic_data.csv')

# For this example, we'll use our sample data generator
data = get_sample_data(n_samples=1000)
print(f"Generated {len(data)} samples.")

# Prepare features and target
print("Preparing features...")
features = [
    'Day_of_Week', 'Hour', 'Weather_Condition', 'Road_Type',
    'Junction_Type', 'Traffic_Volume', 'Speed_Limit'
]

# Convert categorical features to one-hot encoding
data_encoded = pd.get_dummies(
    data, 
    columns=['Day_of_Week', 'Weather_Condition', 'Road_Type', 'Junction_Type']
)

# Select features and target
X = data_encoded.drop(['Timestamp', 'Severity', 'Severity_Numeric', 'Clearance_Time'], axis=1, errors='ignore')
y = data['Severity']  # Using the categorical severity (Low, Medium, High)

print(f"Feature matrix shape: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training Random Forest model...")
# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
print("Saving model to models/random_forest_model.joblib")
joblib.dump(model, 'models/random_forest_model.joblib')

# Save feature columns for reference
pd.Series(X.columns).to_csv('models/feature_columns.csv', index=False)

print("Training complete! You can now use this model in the application.")
print("To use this model, update the initialize_model() function in utils/model.py")
print("See guides/adding_custom_model.md for detailed instructions.")