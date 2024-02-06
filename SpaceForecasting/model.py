from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_and_evaluate_model(df):
    # Select features and target based on actual data
    X = df[['bt_rolling_avg']]  # Example with a single feature. Add more as needed.
    y = df['bt']

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model, X_test, y_test, predictions
