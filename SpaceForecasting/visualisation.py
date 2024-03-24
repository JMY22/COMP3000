import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_forecast(df, forecasts, last_known_time):
    plt.figure(figsize=(15, 7))

    # Determine the period to display actual data
    actual_start_time = last_known_time - pd.Timedelta(days=7)
    actual_end_time = last_known_time

    # Ensure we only plot the period we're interested in for the actual data
    mask = (df.index >= actual_start_time) & (df.index <= actual_end_time)
    actual_data_to_plot = df.loc[mask]

    # Plot the actual data
    plt.plot(actual_data_to_plot.index, actual_data_to_plot['bt'], label='Actual', marker='.', linestyle='-',
             linewidth=1.0)

    # Generate forecast dates starting from the last known time
    forecast_dates = pd.date_range(start=last_known_time, periods=len(forecasts) + 1, freq='min')[1:]

    print(f"Last known time: {last_known_time}")
    print(f"Actual data time range: {df.index.min()} to {df.index.max()}")
    print(f"Forecast dates range: {forecast_dates[0]} to {forecast_dates[-1]}")
    print(f"Actual data sample: {df['bt'].tail(10)}")  # Last 10 actual data points
    print(f"Forecast sample: {forecasts[:10]}")  # First 10 forecasted data points

    # Plot the forecasted data
    plt.plot(forecast_dates, forecasts, label='Forecasted', marker='x', linestyle='--', linewidth=1.0)

    plt.title('Actual vs Forecasted BT')
    plt.xlabel('Time')
    plt.ylabel('BT')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()