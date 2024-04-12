import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def convert_columns_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()


def fetch_solar_wind_data(url):
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
    try:
        new_data = pd.read_csv(csv_file)
        new_data['timedelta'] = pd.to_timedelta(new_data['timedelta'])
        new_data['time_tag'] = reference_timestamp + new_data['timedelta']
        new_data = new_data.sort_values(by='time_tag')
        new_data.set_index('time_tag', inplace=True)
        new_data = new_data[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']]
        return convert_columns_to_numeric(new_data, ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt'])
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading new data: {e}")
        return pd.DataFrame()


def normalize_and_sequence_data(df, n_steps):
    features = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_copy = df.copy()
    df_copy[features] = scaler.fit_transform(df_copy[features])
    X, y = [], []
    for i in range(n_steps, len(df_copy)):
        X.append(df_copy[features].iloc[i - n_steps:i].to_numpy())
        y.append(df_copy[features].iloc[i].to_numpy())
    return np.array(X), np.array(y), scaler
