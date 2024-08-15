import pandas as pd

def clean_data(file_path: str):
    """
    Limpia el conjunto de datos eliminando o imputando valores faltantes.
    
    :param file_path: La ruta del archivo CSV que contiene los datos a limpiar.
    :return: DataFrame limpio.
    """
    data = pd.read_csv(file_path)
    # Eliminar filas con valores nulos
    data = data.dropna()
    return data

if __name__ == "__main__":
    cleaned_data = clean_data('../data/nflx_data.csv')
    cleaned_data.to_csv('../data/nflx_cleaned_data.csv', index=False)
