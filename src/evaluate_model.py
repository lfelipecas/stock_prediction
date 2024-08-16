import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

def evaluate_model(X_test, Y_test, scaler):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'model', 'nflx_lstm_model.h5')
    model = load_model(model_path)

    predicted_prices = model.predict(X_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(Y_test.reshape(-1, 1))

    mse = mean_squared_error(real_prices, predicted_prices)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    return mse, mae, r2, predicted_prices, real_prices

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    model_directory = os.path.join(script_dir, '..', 'model')
    
    X_test = np.load(os.path.join(data_directory, 'X_test.npy'))
    Y_test = np.load(os.path.join(data_directory, 'Y_test.npy'))
    scaler = pd.read_pickle(os.path.join(model_directory, 'scaler.pkl'))

    mse, mae, r2, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    results = pd.DataFrame({
        'Real Prices': real_prices.flatten(),
        'Predicted Prices': predicted_prices.flatten()
    })
    results.to_csv(os.path.join(data_directory, 'evaluation_results.csv'), index=False)
    print(f"Resultados guardados en {os.path.join(data_directory, 'evaluation_results.csv')}")
