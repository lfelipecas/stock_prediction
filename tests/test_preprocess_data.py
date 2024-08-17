import sys  # Import sys to modify the path
import os
import pytest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from unittest.mock import patch

# Adjust the path so Python can find the preprocess_data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess_data import preprocess_data

@pytest.fixture
def mock_data():
    # Create a mock DataFrame to use in tests
    return pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Adj Close': [100, 200, 300]
    })

@patch('joblib.dump')  # Mock joblib.dump to avoid saving files during testing
def test_preprocess_data(mock_joblib_dump, tmpdir, mock_data):
    # Create a temporary file path for input and output
    input_file_path = os.path.join(tmpdir, 'mock_data.csv')
    output_file_path = os.path.join(tmpdir, 'nflx_preprocessed_data.csv')

    # Save the mock data to a temporary CSV file
    mock_data.to_csv(input_file_path, index=False)

    # Run the preprocess_data function with the path to the temporary file
    data_scaled_df, scaler = preprocess_data(input_file_path, output_file_path)

    # Check if the output DataFrame has the expected columns
    assert 'Date' in data_scaled_df.columns
    assert 'Adj Close_scaled' in data_scaled_df.columns

    # Verify the scaling result, ignoring the name attribute of the Series
    expected_scaled_values = MinMaxScaler(feature_range=(0, 1)).fit_transform(mock_data[['Adj Close']])
    pd.testing.assert_series_equal(data_scaled_df['Adj Close_scaled'], pd.Series(expected_scaled_values.flatten()), check_names=False)

    # Check if the scaler was saved
    mock_joblib_dump.assert_called_once()

@patch('joblib.dump')  # Mock joblib.dump to avoid saving files during testing
def test_preprocess_data_scaler_saved(mock_joblib_dump, tmpdir, mock_data):
    # Create a temporary file path for input and output
    input_file_path = os.path.join(tmpdir, 'mock_data.csv')
    output_file_path = os.path.join(tmpdir, 'nflx_preprocessed_data.csv')

    # Save the mock data to a temporary CSV file
    mock_data.to_csv(input_file_path, index=False)

    # Run the preprocess_data function with the path to the temporary file
    _, scaler = preprocess_data(input_file_path, output_file_path)

    # Check if the scaler was saved to the correct path
    expected_scaler_path = os.path.join(os.path.dirname(output_file_path), 'scaler.pkl')
    mock_joblib_dump.assert_called_once_with(scaler, expected_scaler_path)
