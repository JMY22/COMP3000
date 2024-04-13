from data_loader import fetch_solar_wind_data, load_new_data, normalize_and_sequence_data
from model import train_and_save_model, load_and_update_model, forecast, forecast_average_bt
import pandas as pd
from visualisation import plot_forecasts, plot_averages
import os

# Define constants and file paths
MODEL_PATH = 'C:/Users/Joe/Desktop/Kaggle Data/solar_wind_model.keras'
JSON_URL = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
CSV_FILE = 'C:/Users/Joe/Desktop/Kaggle Data/solar_wind.csv'
N_STEPS_SHORT_TERM = 360
N_STEPS_AVERAGE = 1440  # Different n_steps for the daily average calculation
N_FEATURES = 4  # bx_gsm, by_gsm, bz_gsm, bt


def main():
    print("Starting the forecasting process...")

    # Fetching and combining recent and historical data
    df_recent_data = fetch_solar_wind_data(JSON_URL)
    df_historic_data = load_new_data(CSV_FILE, df_recent_data.index.min())
    combined_data = pd.concat([df_historic_data, df_recent_data], axis=0).sort_index()

    # Preparing data for short-term forecasting
    recent_data_subset_short_term = combined_data.tail(
        N_STEPS_SHORT_TERM * 100)  # Using last 100 data points for training
    X_short_term, y_short_term, scaler_short_term = normalize_and_sequence_data(recent_data_subset_short_term,
                                                                                N_STEPS_SHORT_TERM)

    # Split the data, keeping the latest for testing to simulate future unseen data
    train_size = int(len(X_short_term) * 0.8)
    X_train, y_train = X_short_term[:train_size], y_short_term[:train_size]
    X_test, y_test = X_short_term[train_size:], y_short_term[train_size:]

    # Model training/update
    if os.path.exists(MODEL_PATH):
        print("Loading and updating existing model...")
        model = load_and_update_model(MODEL_PATH, X_train, y_train, X_val=X_test, y_val=y_test, epochs=5,
                                      batch_size=128)
    else:
        print("Building and training a new LSTM model...")
        model = train_and_save_model(X_train, y_train, X_val=X_test, y_val=y_test, model_path=MODEL_PATH,
                                     n_features=N_FEATURES, n_steps=N_STEPS_SHORT_TERM, epochs=50, batch_size=128)

    # Short-term forecasting
    print("Model training/update complete. Forecasting the next 200 steps...")
    last_known_sequence = X_test[-1].flatten()
    forecasted_values = forecast(model, scaler_short_term, last_known_sequence, steps=100)
    last_known_time = combined_data.index[-1] + pd.Timedelta(minutes=1)
    forecast_output = []
    for i in range(100):
        forecast_time = last_known_time + pd.Timedelta(minutes=i + 1)
        forecast_data = forecasted_values[i]
        forecast_output.append(f"{forecast_time}, {forecast_data[0]:.2f}, {forecast_data[1]:.2f}, "
                               f"{forecast_data[2]:.2f}, {forecast_data[3]:.2f}")
    print("Forecasted Data for the next 100 steps:")
    print("\n".join(forecast_output))

    # Extract actual 'bt' data for the last 100 minutes for comparison
    actual_bt_data = combined_data['bt'].tail(100)

    # Visualization
    plot_forecasts(actual_bt_data, forecasted_values, last_known_time)

    # Daily average 'bt' forecasting
    print("Forecasting average 'bt' for the next 7 days...")
    recent_data_subset_daily_avg = combined_data.tail(N_STEPS_AVERAGE * 21)
    X_daily_avg, y_daily_avg, scaler_daily_avg = normalize_and_sequence_data(recent_data_subset_daily_avg, N_STEPS_AVERAGE)
    initial_sequence_daily_avg = X_daily_avg[-1].flatten()
    daily_averages_bt = forecast_average_bt(model, scaler_daily_avg, initial_sequence_daily_avg, 7 * N_STEPS_AVERAGE, 1440, N_STEPS_AVERAGE)
    avg_bt_output = [f"Average 'bt' on {pd.to_datetime(combined_data.index[-1]).date() + pd.Timedelta(days=i + 1)}: {avg_bt:.2f}" for i, avg_bt in enumerate(daily_averages_bt)]
    print("\n".join(avg_bt_output))

    # Usage in your main function after forecasting average 'bt'
    past_avg_bt = actual_bt_data.rolling(window=1440).mean().dropna().last('7D').values
    plot_averages(past_avg_bt, daily_averages_bt, pd.to_datetime(combined_data.index[-1]).date())


if __name__ == "__main__":
    main()
