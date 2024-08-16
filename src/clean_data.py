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
    for column in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if column in data.columns:
            if column == 'Date':
                data[column] = pd.to_datetime(data[column], errors='coerce')
            else:
                data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Imputar valores faltantes con la mediana de cada columna
    data.fillna(data.median(), inplace=True)
    
    # Eliminar filas que aún contengan valores nulos después de la imputación
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ruta del archivo CSV original y del archivo limpiado
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')

    # Limpiar los datos y guardarlos en un nuevo archivo CSV
    cleaned_data = clean_data(input_file_path)
    cleaned_data.to_csv(output_file_path, index=False)
    print(f"Archivo guardado exitosamente en {output_file_path}")
