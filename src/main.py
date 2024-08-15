import os
from get_data import download_data
from clean_data import clean_data
from preprocess_data import preprocess_data
from split_data import create_datasets
from train_model import train_model
from evaluate_model import evaluate_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Obtener la ruta del directorio actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir la ruta del directorio de datos y modelo
    data_directory = os.path.join(script_dir, '..', 'data')
    model_directory = os.path.join(script_dir, '..', 'model')
    
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Paso 1: Descargar los datos
    print("Descargando datos...")
    data_file_path = os.path.join(data_directory, 'nflx_data.csv')
    data = download_data("NFLX", "2014-01-01", "2023-12-31")
    data.to_csv(data_file_path)
    print(f"Datos descargados y guardados en {data_file_path}")
    
    # Paso 2: Limpiar los datos
    print("Limpiando datos...")
    cleaned_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')
    cleaned_data = clean_data(data_file_path)
    cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Datos limpios guardados en {cleaned_file_path}")
    
    # Paso 3: Preprocesar los datos
    print("Preprocesando datos...")
    preprocessed_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')
    processed_data, scaler = preprocess_data(cleaned_file_path, preprocessed_file_path)
    scaler_path = os.path.join(model_directory, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)
    print(f"Datos preprocesados guardados en {preprocessed_file_path}")
    
    # Paso 4: Dividir los datos en entrenamiento y prueba
    print("Dividiendo los datos...")
    X, Y = create_datasets(processed_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    np.save(os.path.join(data_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(data_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(data_directory, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_directory, 'Y_test.npy'), Y_test)
    print("Conjuntos de datos guardados exitosamente.")
    
    # Paso 5: Entrenar el modelo
    print("Entrenando el modelo...")
    model = train_model(X_train, Y_train)
    model_path = os.path.join(model_directory, 'nflx_lstm_model.h5')
    model.save(model_path)
    print(f"Modelo guardado exitosamente en {model_path}")
    
    # Paso 6: Evaluar el modelo
    print("Evaluando el modelo...")
    scaler = pd.read_pickle(scaler_path)
    mse, mae, r2, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    
    # Guardar los resultados de la evaluación
    results_df = pd.DataFrame({
        'Real Prices': real_prices.flatten(),
        'Predicted Prices': predicted_prices.flatten()
    })
    results_df.to_csv(os.path.join(data_directory, 'evaluation_results.csv'), index=False)
    print(f"Resultados de la evaluación guardados en {os.path.join(data_directory, 'evaluation_results.csv')}")

if __name__ == "__main__":
    main()
