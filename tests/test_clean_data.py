import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch

# Adjust the path so Python can find the clean_data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from clean_data import clean_data

@pytest.fixture
def mock_csv_data(tmpdir):
    # Create a mock CSV file with sample data for testing
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Adj Close': ['100', '101', 'NaN', '102']
    })
    file_path = tmpdir.join("mock_data.csv")
    data.to_csv(file_path, index=False)
    return str(file_path)

def test_clean_data_conversion(mock_csv_data):
    # Test if 'Adj Close' column is correctly converted to numeric
    cleaned_data = clean_data(mock_csv_data)
    assert pd.api.types.is_numeric_dtype(cleaned_data['Adj Close'])

def test_clean_data_fillna(mock_csv_data):
    # Test if missing values in 'Adj Close' column are filled with the median
    cleaned_data = clean_data(mock_csv_data)
    assert not cleaned_data['Adj Close'].isnull().any()

def test_clean_data_dropna(mock_csv_data):
    # Test if rows with NaN values are dropped (if any still exist after filling)
    cleaned_data = clean_data(mock_csv_data)
    assert cleaned_data.isnull().sum().sum() == 0
