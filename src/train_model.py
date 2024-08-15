import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_datasets(data, n_days=7):
    X, Y = [], []
    for i in range(len(data) - n_days):
        X.append(data[i:i + n_days, 0])
        Y.append(data[i + n_days, 0])
    X, Y = np.array(X), np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, Y

def train_model(X_train, Y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    return model

if __name__ == "__main__":
    data = pd.read_csv('../data/nflx_preprocessed_data.csv').values
    X, Y = create_datasets(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = train_model(X_train, Y_train)
    model.save('../model/nflx_lstm_model.h5')  # Guardar el modelo entrenado
