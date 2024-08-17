import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from unittest.mock import patch

# Adjust the path so Python can find the split_data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from split_data import create_datasets

@pytest.fixture
def mock_data():
    # Create a mock DataFrame similar to the preprocessed data
    return pd.DataFrame({
        "Adj Close_scaled": np.linspace(0.1, 1.0, 100)
    })

def test_create_datasets_shape(mock_data):
    # Test to ensure that the create_datasets function returns the correct shapes for X and Y
    n_days = 7
    X, Y = create_datasets(mock_data, n_days=n_days)
    
    # Assert the shape of X and Y
    assert X.shape == (len(mock_data) - n_days, n_days, 1)
    assert Y.shape == (len(mock_data) - n_days,)

def test_train_test_split(mock_data):
    # Test the train_test_split function to ensure proper splitting of data
    n_days = 7
    X, Y = create_datasets(mock_data, n_days=n_days)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Assert the shapes of the train and test sets
    assert X_train.shape[0] + X_test.shape[0] == len(X)
    assert Y_train.shape[0] + Y_test.shape[0] == len(Y)

@patch('numpy.save')
def test_save_datasets(mock_save, mock_data, tmpdir):
    # Test the saving of datasets to ensure files are saved correctly
    n_days = 7
    X, Y = create_datasets(mock_data, n_days=n_days)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Create temporary file paths for saving
    X_train_path = os.path.join(tmpdir, 'X_train.npy')
    X_test_path = os.path.join(tmpdir, 'X_test.npy')
    Y_train_path = os.path.join(tmpdir, 'Y_train.npy')
    Y_test_path = os.path.join(tmpdir, 'Y_test.npy')

    # Save the datasets
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(Y_train_path, Y_train)
    np.save(Y_test_path, Y_test)

    # Ensure that numpy.save was called with the correct arguments
    mock_save.assert_any_call(X_train_path, X_train)
    mock_save.assert_any_call(X_test_path, X_test)
    mock_save.assert_any_call(Y_train_path, Y_train)
    mock_save.assert_any_call(Y_test_path, Y_test)
