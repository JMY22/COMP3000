from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


def train_and_evaluate_model(df):
    X = df[['bt_rolling_avg']]
    y = df['bt']
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return model, X_test, y_test, predictions


def multi_step_forecast(model, X_last, steps=5):
    forecasts = []
    for _ in range(steps):
        X_next = pd.DataFrame([X_last], columns=['bt_rolling_avg'])
        forecast = model.predict(X_next)[0]
        forecasts.append(forecast)
        X_last = [forecast]
    return forecasts
