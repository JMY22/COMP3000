from data_loader import fetch_solar_wind_data
from model import train_and_evaluate_model, multi_step_forecast
from visualisation import plot_actual_vs_forecast
import pandas as pd

# Load and preprocess your data
json_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
df_solar_wind = fetch_solar_wind_data(json_url)

# Train the model and evaluate its performance
model, X_test, y_test, predictions = train_and_evaluate_model(df_solar_wind)

# Use the last known 'bt_rolling_avg' as the starting point for forecasting
X_last = df_solar_wind['bt_rolling_avg'].iloc[-1]

# Forecast the next 5 steps
forecasts = multi_step_forecast(model, X_last, steps=5)

# Visualize the actual data and forecasts
forecast_start_date = df_solar_wind['time_tag'].iloc[-1] + pd.Timedelta(minutes=1)  # Assuming data is in 1-minute intervals
plot_actual_vs_forecast(df_solar_wind, forecasts, forecast_start_date)

# Assuming `predictions` contains the forecasted values for the test set
print("Forecasted values for the test set:")
print(predictions)

# Assuming `y_test` contains the actual values for the test set
print("Actual values for the test set:")
print(y_test.values)

# If you have future forecasted values beyond the available data
print("Future forecasts:")
print(forecasts)
