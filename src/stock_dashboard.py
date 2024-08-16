import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import time

# Cargar el modelo y el escalador
model_path = 'h:/My Drive/UAO_EIA/5_Desarrollo_Proyectos_IA/stock_prediction/model/nflx_lstm_model.h5'
scaler_path = 'h:/My Drive/UAO_EIA/5_Desarrollo_Proyectos_IA/stock_prediction/model/scaler.pkl'
data_path = 'h:/My Drive/UAO_EIA/5_Desarrollo_Proyectos_IA/stock_prediction/data/nflx_preprocessed_data.csv'

model = load_model(model_path)
scaler = pd.read_pickle(scaler_path)
data = pd.read_csv(data_path)

# Preparar datos para la predicción
n_days = 7  # Número de días en la ventana de predicción
X, Y = [], []
for i in range(len(data) - n_days):
    X.append(data[['Open', 'High', 'Low', 'Close']].iloc[i:i + n_days].values)
    Y.append(data['Close'].iloc[i + n_days])

X = np.array(X)
Y = np.array(Y)

# Predecir precios
predicted_prices = model.predict(X)

# Desescalar precios
predicted_prices = scaler.inverse_transform(np.hstack((predicted_prices, np.zeros((predicted_prices.shape[0], scaler.n_features_in_ - 1)))))[:, 0]
real_prices = scaler.inverse_transform(np.hstack((Y.reshape(-1, 1), np.zeros((Y.shape[0], scaler.n_features_in_ - 1)))))[:, 0]

# Crear un DataFrame para graficar
chart_data = pd.DataFrame({
    "Real Prices": real_prices,
    "Predicted Prices": predicted_prices
})

# Configurar el título de la aplicación
st.title("Stock Price Prediction Dashboard")

# Mostrar gráficos de líneas con animación
st.write("Real and Predicted Prices Over Time")

# Crear la gráfica de líneas
progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(chart_data.iloc[0:1])

for i in range(1, len(chart_data)):
    chart.add_rows(chart_data.iloc[i:i+1])
    status_text.text(f"{int((i+1)/len(chart_data)*100)}% Complete")
    progress_bar.progress(int((i+1)/len(chart_data)*100))
    time.sleep(0.01)

progress_bar.empty()

# Botón para recargar
st.button("Re-run")
