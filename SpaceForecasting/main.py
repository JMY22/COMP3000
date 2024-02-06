from data_loader import fetch_solar_wind_data
from model import train_and_evaluate_model
from visualisation import plot_predictions

if __name__ == "__main__":
    json_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    df_solar_wind = fetch_solar_wind_data(json_url)

    model, X_test, y_test, predictions = train_and_evaluate_model(df_solar_wind)

    plot_predictions(y_test, predictions)
