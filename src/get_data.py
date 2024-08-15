import yfinance as yf
import os

def download_data(ticker: str, start_date: str, end_date: str, interval: str = "1d"):
    """
    Descarga los datos históricos de precios de acciones desde Yahoo Finance.
    
    :param ticker: El símbolo de la acción, por ejemplo, 'NFLX'.
    :param start_date: Fecha de inicio en formato 'YYYY-MM-DD'.
    :param end_date: Fecha de fin en formato 'YYYY-MM-DD'.
    :param interval: Intervalo de tiempo para los datos, por defecto es '1d' (diario).
    :return: DataFrame con los datos históricos de precios.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

if __name__ == "__main__":
    # Obtener la ruta absoluta del directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Definir la ruta del directorio de datos
    data_directory = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    # Descargar los datos y guardarlos en un archivo CSV en el directorio de datos
    data = download_data("NFLX", "2014-01-01", "2023-12-31")
    data.to_csv(os.path.join(data_directory, 'nflx_data.csv'))
