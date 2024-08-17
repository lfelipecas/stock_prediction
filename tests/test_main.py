import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main import main

@pytest.fixture
def setup_mock_environment(tmpdir):
    # Create mock directories and files
    data_dir = tmpdir.mkdir("data")
    model_dir = tmpdir.mkdir("model")
    
    # Mock data file paths
    data_file = data_dir.join("nflx_data.csv")
    cleaned_data_file = data_dir.join("nflx_cleaned_data.csv")
    preprocessed_data_file = data_dir.join("nflx_preprocessed_data.csv")
    scaler_file = model_dir.join("scaler.pkl")
    
    return {
        "data_dir": data_dir,
        "model_dir": model_dir,
        "data_file": data_file,
        "cleaned_data_file": cleaned_data_file,
        "preprocessed_data_file": preprocessed_data_file,
        "scaler_file": scaler_file
    }

@patch('main.download_data')
@patch('main.clean_data')
@patch('main.preprocess_data')
@patch('main.create_datasets')
@patch('main.train_model')
@patch('main.evaluate_model')
@patch('pandas.to_pickle')
def test_main(mock_to_pickle, mock_evaluate_model, mock_train_model, mock_create_datasets, mock_preprocess_data, mock_clean_data, mock_download_data, setup_mock_environment):
    # Mock the return values for each function
    mock_download_data.return_value = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Adj Close": [100, 101]
    })
    mock_clean_data.return_value = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Adj Close": [100, 101]
    })
    mock_preprocess_data.return_value = (pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Adj Close_scaled": [0.1, 0.2]
    }), MagicMock())
    mock_create_datasets.return_value = (np.array([[[0.1]], [[0.2]]]), np.array([0.3, 0.4]))
    mock_train_model.return_value = MagicMock()
    mock_evaluate_model.return_value = (0.01, 0.02, 0.9, np.array([0.3, 0.4]), np.array([0.3, 0.4]))

    # Prevent the mock scaler from being pickled
    mock_to_pickle.return_value = None

    # Run the main function
    main()

    # Check if functions were called the expected number of times
    mock_download_data.assert_called_once()
    mock_clean_data.assert_called_once()
    mock_preprocess_data.assert_called_once()
    mock_create_datasets.assert_called_once()
    mock_train_model.assert_called_once()
    mock_evaluate_model.assert_called_once()

    # Check if the scaler was attempted to be pickled
    # Compare only the filename to avoid issues with differing directories
    expected_filename = os.path.basename(str(setup_mock_environment["scaler_file"]))
    actual_filename = os.path.basename(mock_to_pickle.call_args[0][1])
    assert actual_filename == expected_filename
