import os
import pandas as pd
import numpy as np
from get_data import download_data
from clean_data import clean_data
from preprocess_data import preprocess_data
from split_data import create_datasets
from train_model import train_model
from evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # Define the directories for data and models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    model_directory = os.path.join(script_dir, '..', 'model')
    
    # Create directories if they don't exist
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)
    
    print("Downloading data...")
    data_file_path = os.path.join(data_directory, 'nflx_data.csv')
    data = download_data("NFLX")
    data.to_csv(data_file_path)
    print(f"Data saved to {data_file_path}")
    
    print("Cleaning data...")
    cleaned_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')
    cleaned_data = clean_data(data_file_path)
    cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")
    
    print("Preprocessing data...")
    preprocessed_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')
    processed_data, scaler = preprocess_data(cleaned_file_path, preprocessed_file_path)
    scaler_path = os.path.join(model_directory, 'scaler.pkl')
    pd.to_pickle(scaler, scaler_path)
    print(f"Preprocessed data saved to {preprocessed_file_path}")
    
    print("Splitting the data...")
    X, Y = create_datasets(processed_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    np.save(os.path.join(data_directory, 'X_train.npy'), X_train)
    np.save(os.path.join(data_directory, 'X_test.npy'), X_test)
    np.save(os.path.join(data_directory, 'Y_train.npy'), Y_train)
    np.save(os.path.join(data_directory, 'Y_test.npy'), Y_test)
    print("Datasets saved successfully.")
    
    print("Training the model...")
    model = train_model(X_train, Y_train)
    model_path = os.path.join(model_directory, 'nflx_lstm_model.h5')
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")
    
    print("Evaluating the model...")
    mse, mae, r2, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    
    results_df = pd.DataFrame({
        'Real Prices': real_prices.flatten(),
        'Predicted Prices': predicted_prices.flatten()
    })
    results_df.to_csv(os.path.join(data_directory, 'evaluation_results.csv'), index=False)
    print(f"Evaluation results saved to {os.path.join(data_directory, 'evaluation_results.csv')}")

if __name__ == "__main__":
    main()
