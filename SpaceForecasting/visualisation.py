import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(actual_data, forecasts, last_known_time):
    plt.figure(figsize=(15, 7))

    # Generate forecast dates starting from the last known time
    forecast_dates = pd.date_range(start=last_known_time, periods=len(forecasts), freq='T')

    # Convert forecasts to a pandas Series for easier plotting
    forecast_series = pd.Series(data=[f[3] for f in forecasts], index=forecast_dates)
    # Assuming 'bt' is the last feature index 3

    # Plotting
    plt.plot(actual_data.index, actual_data, label='Actual BT', marker='o', linestyle='-', color='blue', markersize=5)
    plt.plot(forecast_series.index, forecast_series, label='Forecasted BT', marker='x', linestyle='--', color='red',
             markersize=5)

    plt.title('BT Actual vs Forecasted')
    plt.xlabel('Time')
    plt.ylabel('Magnetic Field BT (nT)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
