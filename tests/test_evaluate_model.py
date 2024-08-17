import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Adjust the path so Python can find the evaluate_model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluate_model import evaluate_model

@patch('evaluate_model.load_model')
def test_evaluate_model(mock_load_model):
    # Simulate the model's behavior
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.13], [0.23], [0.33], [0.43]])  # Simulated predictions
    mock_load_model.return_value = mock_model

    # Mock data
    X_test = np.array([[[0.1]], [[0.2]], [[0.3]], [[0.4]]])
    Y_test = np.array([0.12, 0.22, 0.32, 0.42])

    # Call the evaluate_model function
    mse, mae, r2, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler=None)

    # Assert that the model was loaded
    mock_load_model.assert_called_once()

    # Assert that predictions are made
    mock_model.predict.assert_called_once_with(X_test)

    # Verify the metrics are calculated and within expected ranges
    assert mse > 0, f"Expected MSE > 0, got {mse}"
    assert mae > 0, f"Expected MAE > 0, got {mae}"
    assert -1 <= r2 <= 1, f"Expected -1 <= R2 <= 1, got {r2}"

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)
