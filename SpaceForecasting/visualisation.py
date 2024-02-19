import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_forecast(df, forecasts, forecast_start_date):
    plt.figure(figsize=(15, 7))

    # Plot actual data
    plt.plot(df['time_tag'], df['bt'], label='Actual', marker='.', linestyle='-', linewidth=1.0)

    # Prepare forecast timestamps and values for plotting
    forecast_dates = pd.date_range(start=forecast_start_date, periods=len(forecasts),
                                   freq='min')  # Updated 'freq' parameter
    plt.plot(forecast_dates, forecasts, label='Forecasted', marker='x', linestyle='--', linewidth=1.0)

    plt.title('Actual vs Forecasted BT')
    plt.xlabel('Time')
    plt.ylabel('BT')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
