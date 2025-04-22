# Models Directory

This directory is where your trained machine learning models should be stored. The application will look here for custom models to load.

## Supported Models

The Traffic Severity Prediction application supports various model types, as long as they provide the scikit-learn compatible interface:

- A `predict()` method that returns class labels
- A `predict_proba()` method that returns class probabilities

## Default Model

By default, the application uses a Decision Tree model that is trained on synthetic data. This model is created on startup if no custom model is found.

## Adding Your Own Model

To add your own custom model:

1. Train your model using `train_custom_model.py` or your own training script
2. Save the model to this directory using a format like joblib
3. Update the application to use your model by modifying `utils/model.py`

For detailed instructions, see `guides/adding_custom_model.md`

## Recommended File Structure

```
models/
  ├── random_forest_model.joblib     # Your trained model file
  ├── feature_columns.csv            # List of feature columns (for one-hot encoded models)
  └── model_metadata.json            # Optional metadata about your model
```

## Integration Examples

See `examples/custom_model_integration.py` for code examples of how to integrate different model types with the application.