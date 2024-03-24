import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def convert_columns_to_numeric(df, columns):
    """Convert specified columns of a DataFrame to numeric, coercing errors to NaN, and drop rows with NaN."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()


def fetch_solar_wind_data(url):
    """Fetches solar wind data from a given URL, processes it, and returns a cleaned DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        columns = data[0]
        actual_data = data[1:]
        df = pd.DataFrame(actual_data, columns=columns)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df.set_index('time_tag', inplace=True)
        df = df[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']]
        return convert_columns_to_numeric(df, ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt'])
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return pd.DataFrame()


def load_new_data(csv_file, reference_timestamp):
    """Loads new data from a CSV file, adjusts for the reference timestamp, and returns a cleaned DataFrame."""
    try:
        new_data = pd.read_csv(csv_file)
        new_data['timedelta'] = pd.to_timedelta(new_data['timedelta'])
        new_data['time_tag'] = reference_timestamp - new_data['timedelta']
        new_data = new_data.sort_values(by='time_tag')
        new_data.set_index('time_tag', inplace=True)
        new_data = new_data[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']]
        return convert_columns_to_numeric(new_data, ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt'])
    except Exception as e:
        print(f"Error loading new data: {e}")
        return pd.DataFrame()


def normalize_and_sequence_data(df, n_steps):
    """Normalize the dataframe features and create sequences for LSTM model input."""
    features = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df[features].iloc[i-n_steps:i].to_numpy())
        y.append(df[features].iloc[i].to_numpy())
    return np.array(X), np.array(y), scaler
