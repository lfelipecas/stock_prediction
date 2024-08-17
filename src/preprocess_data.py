import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(file_path: str, output_path: str):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Apply MinMax scaling to the 'Adj Close' column
    data['Adj Close_scaled'] = scaler.fit_transform(data[['Adj Close']])

    # Save only the preprocessed column and date
    data_scaled_df = pd.DataFrame({'Date': data['Date'], 'Adj Close_scaled': data['Adj Close_scaled']})
    data_scaled_df.to_csv(output_path, index=False)

    # Save the scaler for future use
    scaler_path = os.path.join(os.path.dirname(output_path), 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    return data_scaled_df, scaler

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_preprocessed_data.csv')

    processed_data, scaler = preprocess_data(input_file_path, output_file_path)

    print(f"Preprocessed data saved to {output_file_path}")
    print(f"Scaler saved to {os.path.join(data_directory, 'scaler.pkl')}")
