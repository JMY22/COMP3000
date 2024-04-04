from data_loader import fetch_solar_wind_data, load_new_data, normalize_and_sequence_data
from model import train_and_save_model, load_and_update_model, forecast, forecast_average_bt
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define constants and file paths
MODEL_PATH = 'C:/Users/Joe/Desktop/Kaggle Data/solar_wind_model.keras'
JSON_URL = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
CSV_FILE = 'C:/Users/Joe/Desktop/Kaggle Data/solar_wind.csv'
N_STEPS = 60
N_FEATURES = 4  # bx_gsm, by_gsm, bz_gsm, bt


def main():
    print("Starting the forecasting process...")

    df_recent_data = fetch_solar_wind_data(JSON_URL)
    df_historic_data = load_new_data(CSV_FILE, df_recent_data.index.min())

    combined_data = pd.concat([df_historic_data, df_recent_data], axis=0).sort_index()
    print(f"Combined data has {combined_data.shape[0]} rows and {combined_data.shape[1]} columns.")

    recent_data_subset = combined_data.tail(10080)
    X, y, scaler = normalize_and_sequence_data(recent_data_subset, N_STEPS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists(MODEL_PATH):
        print("Loading and updating existing model...")
        model = load_and_update_model(MODEL_PATH, X_train, y_train, epochs=5, batch_size=32)
    else:
        print("Building and training a new LSTM model...")
        # Assuming the number of features is the second dimension of X_train
        n_features = X_train.shape[2]
        model, _ = train_and_save_model(X_train, y_train, X_val=X_test, y_val=y_test, model_path=MODEL_PATH,
                                        n_features=n_features, epochs=50, batch_size=32)

    print("Model training/update complete. Forecasting the next step...")
    last_known_sequence = X_test[-1]  # Assuming the last sequence from test set for demo
    forecasted_values = forecast(model, scaler, last_known_sequence, N_FEATURES, steps=1)

    last_known_time = combined_data.index[-1] + pd.Timedelta(minutes=1)
    forecast_output = f"{last_known_time}, {forecasted_values[0][0]:.2f}, {forecasted_values[0][1]:.2f}, {forecasted_values[0][2]:.2f}, {forecasted_values[0][3]:.2f}"
    print("Forecasted Data:")
    print(forecast_output)

    print("Forecasting average 'bt' for the next 7 days...")
    total_steps = 7 * 1440  # 7 days * 1440 minutes per day
    daily_averages_bt = forecast_average_bt(model, scaler, X_test[-1], total_steps, 1440)

    last_known_date = pd.to_datetime(combined_data.index[-1]).date()
    for i, avg_bt in enumerate(daily_averages_bt):
        forecast_date = last_known_date + pd.Timedelta(days=i+1)
        print(f"Average 'bt' on {forecast_date}: {avg_bt:.2f}")


if __name__ == "__main__":
    main()
