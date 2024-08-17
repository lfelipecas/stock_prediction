import sys
import os
import pytest
import pandas as pd
import yfinance as yf
from unittest.mock import patch, MagicMock

# Adjust the path so Python can find the get_data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from get_data import download_data

@pytest.fixture
def mock_yfinance_data():
    # Create a mock DataFrame to simulate the data returned by yfinance
    return pd.DataFrame({
        "Adj Close": [100, 101, 102, 103]
    })

@patch('yfinance.download')
def test_download_data(mock_download, mock_yfinance_data):
    # Test that download_data correctly fetches and returns adjusted close prices with default parameters

    # Configure the mock to return a DataFrame
    mock_download.return_value = mock_yfinance_data

    # Call the download_data function
    data = download_data("NFLX")

    # Check if the data returned is as expected
    pd.testing.assert_frame_equal(data, mock_yfinance_data)
    mock_download.assert_called_once_with("NFLX", start=None, end=None, interval="1d")

@patch('yfinance.download')
def test_download_data_with_dates(mock_download, mock_yfinance_data):
    # Test that download_data correctly handles specific start and end dates

    # Configure the mock to return a DataFrame
    mock_download.return_value = mock_yfinance_data

    # Call the download_data function with specific dates
    start_date = "2023-01-01"
    end_date = "2023-01-04"
    data = download_data("NFLX", start_date=start_date, end_date=end_date)

    # Check if the data returned is as expected
    pd.testing.assert_frame_equal(data, mock_yfinance_data)
    mock_download.assert_called_once_with("NFLX", start=start_date, end=end_date, interval="1d")

@patch('yfinance.download')
def test_download_data_with_interval(mock_download, mock_yfinance_data):
    # Test that download_data correctly handles a custom interval

    # Configure the mock to return a DataFrame
    mock_download.return_value = mock_yfinance_data

    # Call the download_data function with a specific interval
    interval = "1wk"
    data = download_data("NFLX", interval=interval)

    # Check if the data returned is as expected
    pd.testing.assert_frame_equal(data, mock_yfinance_data)
    mock_download.assert_called_once_with("NFLX", start=None, end=None, interval=interval)
