"""
Example of how to integrate a custom model into the Traffic Severity Prediction app.

This file shows the modifications needed in utils/model.py to use a custom trained model.
"""

# Modified initialize_model() function
def initialize_model():
    """
    Load a custom pre-trained Random Forest model.
    
    Note: This is an example of how to replace the existing function
    in utils/model.py with one that loads your custom model.
    """
    import joblib
    
    # Load the saved model
    try:
        model = joblib.load('models/random_forest_model.joblib')
        print("Loaded custom Random Forest model")
        return model
    except FileNotFoundError:
        print("Custom model not found. Using default model.")
        # Fall back to the default model
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        # Train on synthetic data (same as before)
        X, y = _generate_training_data(1000)
        model.fit(X, y)
        return model

# Modified _process_input_features() function for Random Forest model
def _process_input_features(input_data):
    """
    Process input features for a Random Forest model that was trained on one-hot encoded data.
    
    Note: This shows how to modify the processing function to match your custom model's 
    expected input format.
    """
    import pandas as pd
    
    # Create a DataFrame with a single row containing the input data
    df = pd.DataFrame([input_data])
    
    # Convert categorical features to one-hot encoding
    df_encoded = pd.get_dummies(
        df, 
        columns=['Day_of_Week', 'Weather_Condition', 'Road_Type', 'Junction_Type']
    )
    
    # Load the list of expected columns (saved during training)
    try:
        import pandas as pd
        expected_columns = pd.read_csv('models/feature_columns.csv', header=None).iloc[:, 0].tolist()
        
        # Add any missing columns with zeros
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder columns to match the training data
        df_encoded = df_encoded[expected_columns]
        
    except FileNotFoundError:
        # If feature columns file is not found, return the encoded DataFrame as is
        pass
    
    # Return the features as a numpy array
    return df_encoded.values[0]

# Modified _generate_explanation function for Random Forest
def _generate_explanation(model, processed_input, raw_input):
    """
    Generate an explanation for a Random Forest model's prediction.
    
    Note: This shows how to modify the explanation function to work with
    different model types.
    """
    # For Random Forest, we can use feature importances
    # In a real application, you might want to use SHAP values or other methods
    
    try:
        # Get feature importances from the model
        importances = model.feature_importances_
        
        # Get feature names (if using one-hot encoding)
        import pandas as pd
        feature_names = pd.read_csv('models/feature_columns.csv', header=None).iloc[:, 0].tolist()
        
        # Create explanation tuples
        explanation = []
        
        # If using one-hot encoding, we need to group the importance by original feature
        feature_groups = {}
        for i, name in enumerate(feature_names):
            # Extract the original feature name from the one-hot encoded column
            if '_' in name:
                base_feature = name.split('_')[0]
                if base_feature in ['Day', 'Weather', 'Road', 'Junction']:
                    # For categorical features that were one-hot encoded
                    if name in feature_groups:
                        feature_groups[name] += importances[i]
                    else:
                        feature_groups[name] = importances[i]
            else:
                # For numerical features
                feature_groups[name] = importances[i]
        
        # Create explanation for the top features
        for name, importance in sorted(feature_groups.items(), key=lambda x: x[1], reverse=True)[:5]:
            if importance > 0.01:  # Only include significant factors
                # Get the original value for this feature
                if name in raw_input:
                    value = raw_input[name]
                    explanation.append((f"{name}: {value}", f"{importance:.2f}"))
                else:
                    # For one-hot encoded features we might need to reconstruct the original value
                    explanation.append((f"{name}", f"{importance:.2f}"))
        
        return explanation[:3]  # Return top 3 factors
        
    except (AttributeError, FileNotFoundError):
        # Fall back to a simpler explanation if there's an error
        return [
            (f"Weather: {raw_input['Weather_Condition']}", "0.30"),
            (f"Traffic: {raw_input['Traffic_Volume']}", "0.25"),
            (f"Road Type: {raw_input['Road_Type']}", "0.20")
        ]