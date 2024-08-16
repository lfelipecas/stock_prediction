import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo LSTM
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../model/nflx_lstm_model.h5")
model = load_model(model_path)

# Cargar los datos preprocesados
data_path = os.path.join(script_dir, "../data/nflx_preprocessed_data.csv")
data = pd.read_csv(data_path)

# Convertir las fechas a datetime
data['Date'] = pd.to_datetime(data['Date'])

# Predecir precios históricos
def predict_historical_prices(data):
    X = data[["Adj Close_scaled"]].values
    
    # Crear una lista para almacenar las secuencias de 7 días
    X_sequences = []
    
    # Generar las secuencias de 7 días
    for i in range(len(X) - 7):
        X_sequences.append(X[i:i + 7])
    
    # Convertir la lista a un array numpy y darle la forma correcta
    X_sequences = np.array(X_sequences)
    
    predicted_prices = model.predict(X_sequences)
    
    data['Adj Close_scaled_Predicted'] = np.nan
    data.loc[7:, 'Adj Close_scaled_Predicted'] = predicted_prices.flatten()

    return data

data = predict_historical_prices(data)

# Gráfico de rendimiento histórico del modelo
st.subheader("Historical Model Performance")

def plot_historical_performance():
    st.line_chart(data.set_index('Date')[['Adj Close_scaled', 'Adj Close_scaled_Predicted']])

plot_historical_performance()

# Función para preparar la entrada del modelo para predicciones futuras
def prepare_input_for_prediction(last_known_values, timesteps=7):
    if last_known_values.shape[0] < timesteps:
        padding = np.zeros((timesteps - last_known_values.shape[0], 1))
        X_input = np.vstack((padding, last_known_values))
    else:
        X_input = last_known_values[-timesteps:]
    
    X_input = np.reshape(X_input, (1, timesteps, 1))
    return X_input

# Predicciones futuras
st.subheader("Future Predictions")
n_days_future = st.selectbox("Select Prediction Window (days):", [7, 14, 30, 60, 90, 180])

def plot_future_predictions(n_days):
    last_known_values = data[["Adj Close_scaled"]].dropna().values
    X_input = prepare_input_for_prediction(last_known_values, timesteps=7)
    
    future_predictions = []
    for _ in range(n_days):
        pred = model.predict(X_input)[0]
        future_predictions.append(pred)
        X_input = np.append(X_input[:, 1:, :], [[pred]], axis=1)

    future_predictions = np.array(future_predictions)
    future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=n_days + 1)[1:]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Adj Close_scaled Prices': future_predictions.flatten()
    })
    future_df.set_index('Date', inplace=True)
    
    st.line_chart(future_df)

plot_future_predictions(n_days_future)
