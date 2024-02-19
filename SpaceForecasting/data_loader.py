import requests
import pandas as pd


def fetch_solar_wind_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad responses
        data = response.json()

        # The first list contains the column names
        columns = data[0]
        # The rest of the lists contain the data
        actual_data = data[1:]

        # Create DataFrame with the correct column names
        df = pd.DataFrame(actual_data, columns=columns)

        # Converting numeric columns to appropriate data types
        numeric_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt', 'lon_gsm', 'lat_gsm']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parsing 'time_tag' to datetime
        df['time_tag'] = pd.to_datetime(df['time_tag'])

        # Example preprocessing: Calculate rolling averages for magnetic field total strength 'bt'
        df['bt_rolling_avg'] = df['bt'].rolling(window=24, min_periods=1).mean()

        # Optional: Drop rows with NaN values if necessary after preprocessing
        df.dropna(inplace=True)

        print(df.head())  # Display the first few rows to verify the DataFrame
        return df
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error


if __name__ == "__main__":
    json_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    df_solar_wind = fetch_solar_wind_data(json_url)

    output_file_path = 'preprocessed_solar_wind_data.csv'
    df_solar_wind.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved to {output_file_path}")
