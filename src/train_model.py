import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def train_model(X_train, Y_train):
    """
    Trains an LSTM model using the provided training data.

    :param X_train: Training data for the input sequences.
    :param Y_train: Training data for the target values.
    :return: Trained LSTM model.
    """
    # # Ensure that the data is in float32 format
    # X_train = X_train.astype('float32')
    # Y_train = Y_train.astype('float32')

    # # Define the LSTM model architecture
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(LSTM(units=50))
    # model.add(Dense(1))  # Output layer with a single neuron for 'Adj Close' prediction
    # model.compile(optimizer='adam', loss='mean_squared_error')
    
    # # Train the model
    # model.fit(X_train, Y_train, epochs=20, batch_size=32)
    # return model

    # Ensure that the data is in float32 format
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Output layer with a single neuron for 'Adj Close' prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, Y_train, epochs=20, batch_size=32)
    return model

if __name__ == "__main__":
    # Define paths for the script directory and data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    
    # Load the training data
    X_train = np.load(os.path.join(data_directory, 'X_train.npy'), allow_pickle=True)
    Y_train = np.load(os.path.join(data_directory, 'Y_train.npy'), allow_pickle=True)

    # Train the model with the training data
    model = train_model(X_train, Y_train)

    # Define the model directory and save the trained model
    model_directory = os.path.join(script_dir, '..', 'model')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Save the trained model
    model_path = os.path.join(model_directory, 'nflx_lstm_model.h5')
    model.save(model_path)
    print(f"Model successfully saved to {model_path}")
