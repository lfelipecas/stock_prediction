import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import time

# Cargar el modelo y el escalador
model_path = '../model/nflx_lstm_model.h5'
scaler_path = '../model/scaler.pkl'
data_path = '../data/nflx_preprocessed_data.csv'

model = load_model(model_path)
scaler = pd.read_pickle(scaler_path)
data = pd.read_csv(data_path)

# Interfaz de usuario para seleccionar el tipo de precio y la ventana de predicción
st.title("Stock Price Prediction Dashboard")
price_type = st.selectbox("Select Price Type", ["Open", "High", "Low", "Close"])
n_days_option = st.selectbox("Select Prediction Window Size", ["7 Days", "14 Days", "21 Days", "1 Month", "2 Months", "3 Months", "4 Months", "5 Months", "6 Months"])
n_days_mapping = {"7 Days": 7, "14 Days": 14, "21 Days": 21, "1 Month": 30, "2 Months": 60, "3 Months": 90, "4 Months": 120, "5 Months": 150, "6 Months": 180}
n_days = n_days_mapping[n_days_option]

# Preparar los datos históricos con todas las características
def prepare_data(data, n_days):
    X, Y = [], []
    for i in range(len(data) - n_days):
        X.append(data[['Open', 'High', 'Low', 'Close']].iloc[i:i + n_days].values)
        Y.append(data[price_type].iloc[i + n_days])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X, Y = prepare_data(data, n_days)

# Verificar las dimensiones de entrada
st.write(f"X shape (Historical): {X.shape}")
st.write(f"Y shape (Historical): {Y.shape}")

# Predicción de precios históricos
predicted_prices_historical = model.predict(X)

# Desescalar precios históricos
predicted_prices_historical = scaler.inverse_transform(np.hstack((predicted_prices_historical, np.zeros((predicted_prices_historical.shape[0], scaler.n_features_in_ - 1)))))[:, 0]
real_prices = scaler.inverse_transform(np.hstack((Y.reshape(-1, 1), np.zeros((Y.shape[0], scaler.n_features_in_ - 1)))))[:, 0]

# Crear un DataFrame para graficar los precios históricos
historical_chart_data = pd.DataFrame({
    "Real Prices": real_prices,
    "Predicted Prices": predicted_prices_historical
})

# Predicción futura
def predict_future_prices(data, n_days, future_days):
    last_data = data[['Open', 'High', 'Low', 'Close']].iloc[-n_days:].values
    predictions = []
    
    for _ in range(future_days):
        last_data_scaled = scaler.transform(last_data)
        last_data_scaled = last_data_scaled.reshape((1, last_data_scaled.shape[0], last_data_scaled.shape[1]))
        predicted_price = model.predict(last_data_scaled)
        predicted_price_descaled = scaler.inverse_transform(
            np.hstack((predicted_price, np.zeros((predicted_price.shape[0], scaler.n_features_in_ - 1))))
        )[:, 0]
        
        predictions.append(predicted_price_descaled[0])
        
        # Actualizar los datos para la próxima predicción
        new_data = np.append(last_data[1:], [[predicted_price_descaled[0]] * 4], axis=0)
        last_data = new_data
    
    return predictions

# Realizar la predicción futura
future_predictions = predict_future_prices(data, n_days, n_days)

# Crear un DataFrame para graficar las predicciones futuras
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=n_days + 1)[1:]
future_chart_data = pd.DataFrame({
    "Date": future_dates,
    "Predicted Future Prices": future_predictions
})

# Mostrar gráficos de líneas con animación de precios históricos
st.write("Historical Real and Predicted Prices Over Time")

# Crear la gráfica de líneas de precios históricos
progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(historical_chart_data.iloc[0:1])

for i in range(1, len(historical_chart_data)):
    chart.add_rows(historical_chart_data.iloc[i:i+1])
    status_text.text(f"{int((i+1)/len(historical_chart_data)*100)}% Complete")
    progress_bar.progress(int((i+1)/len(historical_chart_data)*100))
    time.sleep(0.01)

progress_bar.empty()

# Mostrar la predicción futura
st.write(f"Predicted {price_type} Prices for the Next {n_days} Days")
st.line_chart(future_chart_data.set_index("Date"))

# Botón para recargar
st.button("Re-run")
