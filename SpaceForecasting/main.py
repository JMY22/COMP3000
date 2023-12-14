from exploring_data import load_and_explore_data
from solar_wind_forecasting import prepare_data, train_linear_regression_model, evaluate_model


def main():
    # Load and explore data
    data_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    data = load_and_explore_data(data_url)

    # Prepare data for training
    features, target_variable = ['by_gsm', 'bz_gsm', 'lon_gsm', 'lat_gsm', 'bt'], 'bx_gsm'
    x_train, x_test, y_train, y_test = prepare_data(data, features, target_variable)

    # Train linear regression model
    model = train_linear_regression_model(x_train, y_train)

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Make predictions on new data (if needed)


if __name__ == "__main__":
    main()
