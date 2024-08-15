import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(X_train, Y_train):
    """
    Entrena el modelo LSTM.
    
    :param X_train: Conjunto de datos de entrenamiento X.
    :param Y_train: Conjunto de datos de entrenamiento Y.
    :return: Modelo entrenado.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    return model

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir la ruta de los conjuntos de datos
    data_directory = os.path.join(script_dir, '..', 'data')
    X_train = np.load(os.path.join(data_directory, 'X_train.npy'), allow_pickle=True)
    Y_train = np.load(os.path.join(data_directory, 'Y_train.npy'), allow_pickle=True)
    
    # Verificar la estructura y los tipos de datos
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Y_train shape: {Y_train.shape}, dtype: {Y_train.dtype}")
    print(f"First few samples of X_train: {X_train[:5]}")
    print(f"First few samples of Y_train: {Y_train[:5]}")

    # Asegurarse de que los datos sean del tipo correcto
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    
    # Entrenar el modelo
    model = train_model(X_train, Y_train)
    
    # Guardar el modelo entrenado
    model_directory = os.path.join(script_dir, '..', 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    model_path = os.path.join(model_directory, 'nflx_lstm_model.h5')
    model.save(model_path)
    print(f"Modelo guardado exitosamente en {model_path}")
