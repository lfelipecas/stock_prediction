import pandas as pd
import os

def clean_data(file_path: str):
    data = pd.read_csv(file_path)
    
    # Convert 'Adj Close' column to numeric and handle any errors
    data['Adj Close'] = pd.to_numeric(data['Adj Close'], errors='coerce')
    
    # Fill missing values with the median only in the 'Adj Close' column
    data['Adj Close'] = data['Adj Close'].fillna(data['Adj Close'].median())
    
    # Drop any remaining rows that still have NaN values (shouldn't be any after imputation)
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, '..', 'data')
    input_file_path = os.path.join(data_directory, 'nflx_data.csv')
    output_file_path = os.path.join(data_directory, 'nflx_cleaned_data.csv')

    cleaned_data = clean_data(input_file_path)
    cleaned_data.to_csv(output_file_path, index=False)
    print(f"File saved to {output_file_path}")
