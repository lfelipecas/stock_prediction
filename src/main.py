from src.get_data import download_data
from src.clean_data import clean_data
from src.preprocess_data import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    # 1. Obtener los datos
    data = download_data("NFLX", "2014-01-01", "2023-12-31")
    data.to_csv('data/nflx_data.csv')

    # 2. Limpiar los datos
    cleaned_data = clean_data('data/nflx_data.csv')
    cleaned_data.to_csv('data/nflx_cleaned_data.csv', index=False)

    # 3. Preprocesar los datos
    processed_data, scaler = preprocess_data('data/nflx_cleaned_data.csv')
    pd.DataFrame(processed_data, columns=['Close']).to_csv('data/nflx_preprocessed_data.csv', index=False)

    # 4. Entrenar el modelo
    data = processed_data
    X, Y = create_datasets(data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = train_model(X_train, Y_train)
    model.save('model/nflx_lstm_model.h5')

    # 5. Evaluar el modelo
    mse, predicted_prices, real_prices = evaluate_model(X_test, Y_test, scaler)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()
