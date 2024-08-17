import yfinance as yf
import os

def download_data(ticker: str, start_date: str = None, end_date: str = None, interval: str = "1d"):
    """
    Downloads historical stock price data from Yahoo Finance.

    :param ticker: The stock symbol, e.g., 'NFLX'.
    :param start_date: Start date in 'YYYY-MM-DD' format. If not specified, the earliest available data is retrieved.
    :param end_date: End date in 'YYYY-MM-DD' format. If not specified, the most recent data is retrieved.
    :param interval: Time interval for the data, default is '1d' (daily).
    :return: DataFrame with historical price data.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[['Adj Close']]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    data = download_data("NFLX")  # Download all available data
    data.to_csv(os.path.join(data_directory, 'nflx_data.csv'))

    print(f"Data saved in {os.path.join(data_directory, 'nflx_data.csv')}")
