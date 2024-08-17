import sys
import os
import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from unittest.mock import patch, MagicMock

# Adjust the path so Python can find the train_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train_model import train_model

@pytest.fixture
def mock_training_data():
    # Create mock training data with the appropriate shape
    X_train = np.random.rand(100, 7, 1).astype('float32')
    Y_train = np.random.rand(100, 1).astype('float32')
    return X_train, Y_train

@patch('tensorflow.keras.models.Sequential.fit')
def test_train_model(mock_fit, mock_training_data):
    # Mock the fit method to avoid actual training during testing
    mock_fit.return_value = None

    # Unpack the mock training data
    X_train, Y_train = mock_training_data

    # Train the model
    model = train_model(X_train, Y_train)

    # Check if the model is an instance of Sequential
    assert isinstance(model, Sequential)

    # Ensure the model has the correct architecture
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], LSTM)
    assert isinstance(model.layers[1], LSTM)
    assert isinstance(model.layers[2], Dense)

    # Verify that the fit method was called once
    mock_fit.assert_called_once()
