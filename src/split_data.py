import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_datasets(data, n_days=7):
    """
    Creates datasets for training and testing the model.

    :param data: DataFrame containing the preprocessed data.
    :param n_days: Number of days to consider for creating input sequences.
    :return: X, Y numpy arrays where X is the input sequence and Y is the target value.
    """
    X, Y = [], []
    for i in range(len(data) - n_days):
        # Create sequences of n_days and corresponding target values
        X.append(data.iloc[i:i + n_days]['Adj Close_scaled'].values)
        Y.append(data.iloc[i + n_days]['Adj Close_scaled'])
    
    # Convert lists to numpy arrays
    X, Y = np.array(X), np.array(Y)
    
    # Reshape X for compatibility with LSTM (n_samples, n_days, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, Y

if __name__ == "__main__":
    # Define paths for the script directory and data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    
    # Load the preprocessed data
    input_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')
    data = pd.read_csv(input_file_path)

    # Create datasets using the preprocessed data
    X, Y = create_datasets(data)

    # Split the datasets into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Save the datasets as .npy files
    np.save(os.path.join(data_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(data_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(data_directory, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_directory, 'Y_test.npy'), Y_test)

    # Confirm that the datasets have been saved successfully
    print("Datasets saved successfully.")
