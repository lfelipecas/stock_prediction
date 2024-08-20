# Stock Price Prediction with LSTM

## Introduction
This project focuses on predicting stock prices using Long Short-Term Memory (LSTM) networks. The goal is to build a robust model that can forecast future stock prices based on historical data. The project is particularly suited for data scientists, machine learning enthusiasts, and financial analysts interested in applying deep learning to time series forecasting.

## Project Structure
The project is organized into several directories and files, each serving a specific purpose:

├── data/
│   ├── nflx_data.csv
│   ├── nflx_cleaned_data.csv
│   ├── nflx_preprocessed_data.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── Y_train.npy
│   ├── Y_test.npy
├── model/
│   ├── nflx_lstm_model.h5
│   ├── scaler.pkl
├── src/
│   ├── clean_data.py
│   ├── evaluate_model.py
│   ├── get_data.py
│   ├── main.py
│   ├── preprocess_data.py
│   ├── split_data.py
│   ├── stock_dashboard.py
│   ├── train_model.py
├── tests/
│   ├── test_evaluate_model.py
├── requirements.txt
├── README.md

## Installation
To set up this project, follow the steps below:

1. **Clone the repository**:
   ```
   git clone <repository_url>
   cd stock_prediction
   ```

2. **Install dependencies**:
   Ensure you are using Python 3.9.19. Install the necessary libraries using:
   ```
   pip install -r requirements.txt
   ```

## Description of the Scripts in `src/`

### get_data.py
This script downloads historical stock price data from Yahoo Finance using the `yfinance` library. It retrieves the adjusted closing prices and saves them in a CSV file. 

- **Usage**: Run `get_data.py` to download the data for a specified stock ticker (e.g., NFLX).
- **Output**: `nflx_data.csv` in the `data/` directory.

### clean_data.py
The purpose of this script is to clean the raw stock price data. It converts the 'Adj Close' column to numeric, fills missing values with the median, and removes any remaining rows with missing values.

- **Usage**: Run `clean_data.py` to clean the data in `nflx_data.csv`.
- **Output**: `nflx_cleaned_data.csv` in the `data/` directory.

### preprocess_data.py
This script preprocesses the cleaned data by applying MinMax scaling to the 'Adj Close' column. The scaled data is essential for LSTM models, which perform better with normalized input.

- **Usage**: Run `preprocess_data.py` to scale the data.
- **Output**: `nflx_preprocessed_data.csv` in the `data/` directory, and `scaler.pkl` in the `model/` directory.

### split_data.py
This script splits the preprocessed data into sequences suitable for training and testing an LSTM model. It creates datasets with a specified number of time steps (e.g., 7 days).

- **Usage**: Run `split_data.py` to generate the training and testing datasets.
- **Output**: `X_train.npy`, `X_test.npy`, `Y_train.npy`, and `Y_test.npy` in the `data/` directory.

### train_model.py
This script defines and trains an LSTM model using the prepared training data. The trained model is saved for future use in predictions.

- **Usage**: Run `train_model.py` to train the LSTM model.
- **Output**: `nflx_lstm_model.h5` in the `model/` directory.

### evaluate_model.py
This script evaluates the trained model using the test dataset. It calculates performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score. 

- **Usage**: Run `evaluate_model.py` to evaluate the model's performance.
- **Output**: Metrics displayed in the console and saved predictions in `evaluation_results.csv`.

### main.py
The `main.py` script orchestrates the entire process, from downloading the data to evaluating the trained model. It sequentially executes the steps to prepare data, train the model, and evaluate its performance.

- **Usage**: Run `main.py` to execute the full workflow of data processing, model training, and evaluation.
- **Output**: All intermediary and final outputs as described in the previous scripts.

### stock_dashboard.py
This script creates an interactive dashboard using Streamlit to visualize historical and future stock price predictions. The dashboard allows users to see the performance of the model and to make future predictions over various timeframes.

- **Usage**: Run `stock_dashboard.py` to launch the dashboard.
- **Features**: Visualization of historical model performance, future predictions.

## Usage Guide

### Running the Main Script
To execute the entire pipeline, simply run:
```
python src/main.py
```
This will download data, clean it, preprocess it, split it into training/testing sets, train the model, and evaluate the model's performance.

### Generating CSV Files
All necessary CSV files will be automatically generated in the `data/` directory as you run the scripts. These include cleaned, preprocessed data, and the evaluation results.

### Creating and Using Models
The LSTM model is trained using the `train_model.py` script and saved as `nflx_lstm_model.h5`. The scaler used for data preprocessing is saved as `scaler.pkl`. These files are used during evaluation and prediction.

### Using the Dashboard
To interact with the predictions and visualize them, run the following command to start the Streamlit dashboard:
```
streamlit run src/stock_dashboard.py
```