import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path: str):
    """
    Normaliza los datos y los prepara para el entrenamiento del modelo.
    
    :param file_path: La ruta del archivo CSV con los datos limpios.
    :return: DataFrame normalizado.
    """
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])
    return data_scaled, scaler

if __name__ == "__main__":
    processed_data, scaler = preprocess_data('../data/nflx_cleaned_data.csv')
    pd.DataFrame(processed_data, columns=['Close']).to_csv('../data/nflx_preprocessed_data.csv', index=False)
