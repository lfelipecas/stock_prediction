import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_datasets(data, n_days=7):
    X, Y = [], []
    for i in range(len(data) - n_days):
        X.append(data.iloc[i:i + n_days]['Adj Close_scaled'].values)
        Y.append(data.iloc[i + n_days]['Adj Close_scaled'])
    X, Y = np.array(X), np.array(Y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape para 3D (n_samples, n_days, 1)
    return X, Y

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')

    data = pd.read_csv(input_file_path)

    X, Y = create_datasets(data)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    np.save(os.path.join(data_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(data_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(data_directory, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_directory, 'Y_test.npy'), Y_test)

    print("Conjuntos de datos guardados exitosamente.")
