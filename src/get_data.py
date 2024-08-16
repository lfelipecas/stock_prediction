import yfinance as yf
import os

def download_data(ticker: str, start_date: str = None, end_date: str = None, interval: str = "1d"):
    """
    Descarga los datos históricos de precios de acciones desde Yahoo Finance.
    
    :param ticker: El símbolo de la acción, por ejemplo, 'NFLX'.
    :param start_date: Fecha de inicio en formato 'YYYY-MM-DD'. Si no se especifica, se obtiene el dato más antiguo disponible.
    :param end_date: Fecha de fin en formato 'YYYY-MM-DD'. Si no se especifica, se obtiene el dato más reciente disponible.
    :param interval: Intervalo de tiempo para los datos, por defecto es '1d' (diario).
    :return: DataFrame con los datos históricos de precios.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[['Adj Close']]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    data = download_data("NFLX")  # Descargar todos los datos disponibles
    data.to_csv(os.path.join(data_directory, 'nflx_data.csv'))

    print(f"Datos guardados en {os.path.join(data_directory, 'nflx_data.csv')}")
