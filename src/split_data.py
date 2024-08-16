import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_datasets(data, n_days=7):
    """
    Crea conjuntos de datos X e Y para el modelo.
    
    :param data: DataFrame con los datos preprocesados.
    :param n_days: Número de días para usar en la ventana de predicción.
    :return: Conjuntos X e Y para el modelo.
    """
    # Eliminar la columna 'Date' si existe
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])
    
    X, Y = [], []
    for i in range(len(data) - n_days):
        X.append(data.iloc[i:i + n_days].values)
        Y.append(data.iloc[i + n_days, 0])  # Suponiendo que 'Close' es la primera columna
    X, Y = np.array(X), np.array(Y)
    return X, Y

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir la ruta del archivo preprocesado
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')
    
    # Cargar los datos preprocesados
    data = pd.read_csv(input_file_path)
    
    # Parámetro: número de días en la ventana de predicción
    n_days = 7  # Este valor puede ser modificado desde la interfaz Streamlit
    
    # Crear los conjuntos de datos
    X, Y = create_datasets(data, n_days=n_days)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Guardar los conjuntos de datos para el entrenamiento
    np.save(os.path.join(data_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(data_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(data_directory, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_directory, 'Y_test.npy'), Y_test)
    
    print("Conjuntos de datos guardados exitosamente.")
