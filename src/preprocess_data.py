import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path: str, output_path: str):
    """
    Normaliza los datos y los prepara para el entrenamiento del modelo.
    
    :param file_path: La ruta del archivo CSV con los datos limpios.
    :param output_path: La ruta donde se guardará el archivo CSV preprocesado.
    :return: DataFrame normalizado, escalador utilizado.
    """
    # Cargar los datos
    data = pd.read_csv(file_path)
    
    # Seleccionar las columnas a escalar
    columns_to_scale = ['Open', 'High', 'Low', 'Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[columns_to_scale])
    
    # Crear un DataFrame con las columnas escaladas
    data_scaled_df = pd.DataFrame(data_scaled, columns=columns_to_scale)
    
    # Agregar cualquier otra columna que no se escaló, si es necesario
    if 'Date' in data.columns:
        data_scaled_df['Date'] = data['Date']
    
    # Guardar los datos preprocesados
    data_scaled_df.to_csv(output_path, index=False)
    
    return data_scaled_df, scaler

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Definir las rutas de entrada y salida
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')
    
    # Preprocesar los datos
    processed_data, scaler = preprocess_data(input_file_path, output_file_path)
    
    # Guardar el escalador para uso futuro
    scaler_path = os.path.join(script_dir, '..', 'model', 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)
    print(f"Datos preprocesados guardados en {output_path}")
    print(f"Escalador guardado en {scaler_path}")
