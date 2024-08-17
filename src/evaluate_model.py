import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

def evaluate_model(X_test, Y_test, scaler=None):
    # Load the trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'model', 'nflx_lstm_model.h5')
    model = load_model(model_path)

    # Predict prices using the model
    predicted_prices = model.predict(X_test)

    # Inverse transform the predicted and real prices to get actual values, if scaler is provided
    if scaler is not None:
        predicted_prices = scaler.inverse_transform(predicted_prices)
        real_prices = scaler.inverse_transform(Y_test.reshape(-1, 1))
    else:
        real_prices = Y_test

    # Calculate metrics
    mse = mean_squared_error(real_prices, predicted_prices)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    return mse, mae, r2, predicted_prices, real_prices
