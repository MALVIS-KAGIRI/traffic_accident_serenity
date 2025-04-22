# Adding a Custom Model to the Traffic Severity Prediction App

This guide explains how to train and integrate your own custom machine learning model into the Traffic Severity Prediction application. By following these steps, you can replace the default decision tree model with your own trained model.

## Step 1: Train Your Custom Model

First, you need to train your machine learning model. You can use the provided sample script `train_custom_model.py` as a template. This script demonstrates:

1. How to generate or load training data
2. How to train a machine learning model (Random Forest in the example)
3. How to save the trained model and its required metadata

To train a model using the sample script:

```bash
python train_custom_model.py
```

This will:
- Generate synthetic training data based on traffic incident patterns
- Train a Random Forest model on this data
- Save the model to `models/random_forest_model.joblib`
- Save the feature columns list to `models/feature_columns.csv` (important for one-hot encoded features)

## Step 2: Customize the Model Integration

After training your model, you need to modify how the application loads and uses it. The key file to modify is `utils/model.py`.

### For a Simple Model Swap

If your model has the same interface as scikit-learn models (with `.predict()` and `.predict_proba()` methods), you can simply update the `initialize_model()` function in `utils/model.py`:

```python
def initialize_model():
    """
    Load a custom pre-trained model.
    """
    import joblib
    
    # Load the saved model
    try:
        model = joblib.load('models/random_forest_model.joblib')
        print("Loaded custom model")
        return model
    except FileNotFoundError:
        print("Custom model not found. Using default model.")
        # Fall back to the default model
        # [default model code]
```

### For More Complex Models

If your model requires different processing for input features or generates explanations differently, you'll need to modify the following functions:

1. `_process_input_features()` - Handles feature engineering and preprocessing
2. `_generate_explanation()` - Creates explanations for the model's predictions

See `examples/custom_model_integration.py` for a complete example of these modifications.

## Step 3: Testing Your Custom Model

After integrating your model:

1. Restart the application server
2. Navigate to the Prediction page
3. Make a test prediction to ensure your model is working correctly
4. Check the Dashboard and Model Performance pages to see metrics about your model

## Advanced Customizations

### Using Different Model Types

You can integrate models from any Python machine learning library, including:

- scikit-learn models (RandomForest, SVM, etc.)
- XGBoost, LightGBM or other gradient boosting libraries
- TensorFlow or PyTorch deep learning models
- Custom-built models

Just make sure your model wrapper provides the expected interface:

- A `predict()` method that returns class labels
- A `predict_proba()` method that returns class probabilities
- (Optional) A `feature_importances_` attribute for explanations

### Working with External Data

If you want to train your model on real traffic data:

1. Prepare your dataset in CSV format with columns matching the application's expected features
2. Modify the training script to load your data instead of generating synthetic data
3. Ensure your features are properly aligned with what the application expects

## Need More Help?

Consult the following resources:

- `examples/custom_model_integration.py` - Complete example of model integration
- `train_custom_model.py` - Example training script
- scikit-learn documentation for model persistence: [https://scikit-learn.org/stable/model_persistence.html](https://scikit-learn.org/stable/model_persistence.html)