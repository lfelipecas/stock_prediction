import pandas as pd
import os

def clean_data(file_path: str):
    """
    Limpia el conjunto de datos verificando tipos, eliminando o imputando valores faltantes.
    
    :param file_path: La ruta del archivo CSV que contiene los datos a limpiar.
    :return: DataFrame limpio.
    """
    # Cargar los datos
    data = pd.read_csv(file_path)
    
    # Verificación y conversión de tipos de datos
    expected_types = {
        'Date': 'datetime64',
        'Open': 'float',
        'High': 'float',
        'Low': 'float',
        'Close': 'float',
        'Adj Close': 'float',
        'Volume': 'float'
    }

    for column, expected_type in expected_types.items():
        if column in data.columns:
            if expected_type == 'datetime64':
                data[column] = pd.to_datetime(data[column], errors='coerce')
            else:
                data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Imputar valores faltantes con la mediana de cada columna
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)
    
    # Eliminar filas que aún contengan valores nulos después de la imputación
    data = data.dropna()
    
    # Depuración: Verificar el contenido del DataFrame
    print("DataFrame después de la limpieza:")
    print(data.head())
    print(f"Total de filas: {len(data)}")
    
    return data

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ruta del archivo CSV original y del archivo limpiado
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')

    # Verificar que el directorio de datos existe
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Limpiar los datos y guardarlos en un nuevo archivo CSV
    cleaned_data = clean_data(input_file_path)
    cleaned_data.to_csv(output_file_path, index=False)
    print(f"Archivo guardado exitosamente en {output_file_path}")
