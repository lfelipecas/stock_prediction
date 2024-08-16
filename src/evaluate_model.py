import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

def evaluate_model(X_test, Y_test, scaler):
    """
    Evalúa el modelo LSTM en el conjunto de datos de prueba.
    
    :param X_test: Conjunto de datos de prueba X.
    :param Y_test: Conjunto de datos de prueba Y.
    :param scaler: Escalador usado para la normalización de los datos.
    :return: MSE, MAE, R2 y las predicciones junto con los valores reales.
    """
    # Cargar el modelo guardado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'model', 'nflx_lstm_model.h5')
    model = load_model(model_path)

    # Realizar predicciones
    predicted_prices = model.predict(X_test)

    # Invertir la normalización para obtener precios reales
    predicted_prices = scaler.inverse_transform(
        np.hstack((predicted_prices, np.zeros((predicted_prices.shape[0], scaler.n_features_in_ - 1))))
    )[:, 0]
    real_prices = scaler.inverse_transform(
        np.hstack((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], scaler.n_features_in_ - 1))))
    )[:, 0]

    # Calcular métricas de evaluación
    mse = mean_squared_error(real_prices, predicted_prices)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)

    return mse, mae, r2, predicted_prices, real_prices

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir la ruta de los conjuntos de datos y del escalador
    data_directory = os.path.join(script_dir, '..', 'data')
    model_directory = os.path.join(script_dir, '..', 'model')
    
    X_test = np.load(os.path.join(data_directory, 'X_test.npy'))
    Y_test = np.load(os.path.join(data_directory, 'Y_test.npy'))
    scaler = pd.read_pickle(os.path.join(model_directory, 'scaler.pkl'))

    # Evaluar el modelo
    mse, mae, r2, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)

    # Mostrar los resultados
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")

    # Guardar los resultados de la evaluación en un archivo CSV para análisis adicional
    results = pd.DataFrame({
        'Real Prices': real_prices.flatten(),
        'Predicted Prices': predicted_prices.flatten()
    })
    results.to_csv(os.path.join(data_directory, 'evaluation_results.csv'), index=False)
    print(f"Resultados de la evaluación guardados en {os.path.join(data_directory, 'evaluation_results.csv')}")
