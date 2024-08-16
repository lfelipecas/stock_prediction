import pandas as pd
import os

def clean_data(file_path: str):
    data = pd.read_csv(file_path)
    
    # Convertir la columna 'Adj Close' a numérico y manejar cualquier error
    data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
    
    # Rellenar valores faltantes con la mediana solo en la columna 'Adj Close'
    data['Adj Close'].fillna(data['Adj Close'].median(), inplace=True)
    
    # Eliminar cualquier fila restante que aún tenga valores NaN (no debería haber ninguna después de la imputación)
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')

    cleaned_data = clean_data(input_file_path)
    cleaned_data.to_csv(output_file_path, index=False)
    print(f"Archivo guardado en {output_file_path}")
