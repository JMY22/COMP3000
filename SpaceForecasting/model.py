from keras.src.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np


def build_and_train_model(X_train, y_train, X_val, y_val, n_features):
    """
    Builds and trains an LSTM model for forecasting all features.

    Args:
    - X_train: Training data features.
    - y_train: Training data target.
    - X_val: Validation data features.
    - y_val: Validation data target.
    - n_features: Number of features in the dataset.

    Returns:
    - model: The trained LSTM model.
    - history: Training history object.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], n_features)),
        Dense(n_features)  # Predicting all features simultaneously
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=1)

    return model, history


def multi_step_forecast(model, scaler, last_known_sequence, n_features, steps=1):
    """
    Forecast the next 'steps' observations given the last known sequence.

    Args:
    - model: The trained LSTM model.
    - scaler: Scaler object used to inverse transform the forecasted data.
    - last_known_sequence: The last known data sequence to start forecasting from.
    - n_features: Number of features in the dataset.
    - steps: Number of future steps to forecast.

    Returns:
    - forecasts: The forecasted data points.
    """
    forecasts = []
    current_sequence = last_known_sequence.copy()

    for _ in range(steps):
        # Reshape the current sequence for prediction
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))
        # Predict the next time step
        forecasted_step = model.predict(current_sequence_reshaped)[0]
        # Append the forecasted step
        forecasts.append(forecasted_step)
        # Update the current sequence
        current_sequence = np.vstack([current_sequence[1:], forecasted_step])

    # Inverse transform the forecasts to the original scale
    forecasts = scaler.inverse_transform(forecasts)
    return forecasts
