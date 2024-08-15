import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

def evaluate_model(X_test, Y_test, scaler):
    model = load_model('../model/nflx_lstm_model.h5')
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(Y_test.reshape(-1, 1))
    mse = mean_squared_error(real_prices, predicted_prices)
    return mse, predicted_prices, real_prices

if __name__ == "__main__":
    scaler = ...  # Aquí debes cargar el escalador que guardaste previamente
    data = pd.read_csv('../data/nflx_preprocessed_data.csv').values
    X, Y = create_datasets(data)
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    mse, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)
    print(f"Mean Squared Error: {mse}")
