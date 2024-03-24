from data_loader import fetch_solar_wind_data, load_new_data, normalize_and_sequence_data
from model import build_and_train_model, multi_step_forecast
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Starting the forecasting process...")

    # Define the URLs and file paths
    json_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    csv_file = 'C:/Users/Joe/Desktop/Kaggle Data/solar_wind.csv'  # Update with your CSV file path
    print("Defined URLs and file paths.")

    # Fetch recent JSON data
    print("Fetching recent solar wind data...")
    df_recent_data = fetch_solar_wind_data(json_url)
    reference_timestamp = df_recent_data.index.min()

    # Load historic data
    print("Loading historic solar wind data...")
    df_historic_data = load_new_data(csv_file, reference_timestamp)

    # Combine historic and recent data, sort chronologically
    print("Combining historic and current data...")
    combined_data = pd.concat([df_historic_data, df_recent_data], axis=0).sort_index()
    print(f"Combined data has {combined_data.shape[0]} rows and {combined_data.shape[1]} columns.")

    # Instead of using the entire combined dataset, select a recent subset
    print("Selecting a recent subset of the data for training...")
    subset_size = 10000  # For example, the most recent 30,000 rows
    recent_data_subset = combined_data.tail(subset_size)

    # Prepare data for LSTM model with the subset
    print("Preparing data for the LSTM model with the recent data subset...")
    n_steps = 60  # Adjust based on your preference and experimentation
    X, y, scaler = normalize_and_sequence_data(recent_data_subset, n_steps)
    print("Data preparation complete.")

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    print("Building and training the LSTM model...")
    n_features = 4  # Number of features (bx_gsm, by_gsm, bz_gsm, bt)
    model, history = build_and_train_model(X_train, y_train, X_test, y_test, n_features)
    print("Model training complete.")

    # Forecasting the next step
    print("Forecasting the next step...")
    forecasts = multi_step_forecast(model, scaler, X_test[-1], n_features, steps=1)
    print("Forecasting complete.")

    # Reversing normalization to obtain the original scale of forecasted values
    last_known_time = combined_data.index[-1] + pd.Timedelta(minutes=1)  # Assuming 1-minute intervals
    forecasted_values = forecasts[0]  # Extracting the single step forecast

    # Formatting the forecast output
    forecast_output = f"{last_known_time}, {forecasted_values[0]:.2f}, {forecasted_values[1]:.2f}, {forecasted_values[2]:.2f}, {forecasted_values[3]:.2f}"
    print("Forecasted Data:")
    print(forecast_output)
