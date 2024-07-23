import pandas as pd
import numpy as np
import joblib


def load_transformers(models_folder):
    numeric_transformer = joblib.load(
        models_folder + 'numerical_scaler.joblib'
    )
    categorical_transformer = joblib.load(
        models_folder + 'categorical_encoder.joblib'
    )
    return numeric_transformer, categorical_transformer


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    models_folder = '../models/'
    numeric_transformer, categorical_transformer = load_transformers(
        models_folder
    )

    # Load the model
    model = joblib.load(models_folder + 'model.joblib')

    # Preprocess input data

    X = input_data[['price', 'area', 'bedrooms', 'bathrooms', 'stories',
                    'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'parking', 'prefarea',
                    'furnishingstatus']]

    X_numeric = numeric_transformer.transform(X[['area', 'bedrooms']])
    X_categorical = categorical_transformer.transform(X[['mainroad',
                                                        'furnishingstatus']])
    X_final = np.concatenate([X_numeric, X_categorical], axis=1)

    # Make predictions
    predictions = model.predict(X_final)
    return predictions
