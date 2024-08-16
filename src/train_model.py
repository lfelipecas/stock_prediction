import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(X_train, Y_train):
    # Asegurarse de que los datos estén en formato float32
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Cambiado para predecir solo 'Adj Close'
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    X_train = np.load(os.path.join(data_directory, 'X_train.npy'), allow_pickle=True)
    Y_train = np.load(os.path.join(data_directory, 'Y_train.npy'), allow_pickle=True)

    model = train_model(X_train, Y_train)

    model_directory = os.path.join(script_dir, '..', 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_path = os.path.join(model_directory, 'nflx_lstm_model.h5')
    model.save(model_path)
    print(f"Modelo guardado exitosamente en {model_path}")
