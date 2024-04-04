from keras.src.callbacks import EarlyStopping
from keras.src.saving import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np


def build_model(n_features, n_steps=60, output_units=50, dropout_rate=0.2):
    """
    Builds an LSTM model with dropout for regularization.
    """
    model = Sequential([
        LSTM(output_units, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(output_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_save_model(X_train, y_train, X_val, y_val, model_path, n_features, n_steps=60, epochs=100, batch_size=32):
    """
    Trains the LSTM model and saves it to the specified path.
    """
    model = build_model(n_features, n_steps)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        callbacks=[early_stopping])
    model.save(model_path)
    return model, history


def load_and_update_model(model_path, X, y, epochs=100, batch_size=96):
    """
    Loads an existing model and updates it with new data.
    """
    model = load_model(model_path)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    model.save(model_path)
    return model


def forecast(model, scaler, last_known_sequence, n_features, steps=1):
    """
    Forecasts future steps based on the last known sequence.
    """
    forecasts = []
    current_sequence = last_known_sequence.reshape((1, last_known_sequence.shape[0], n_features))

    for _ in range(steps):
        forecasted_step = model.predict(current_sequence)[0]
        forecasts.append(forecasted_step)
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = forecasted_step

    forecasts = np.array(forecasts)
    # Assuming scaler is fitted on the dataset with the same number of features
    forecasts = scaler.inverse_transform(forecasts)
    return forecasts


def forecast_average_bt(model, scaler, initial_sequence, total_steps, steps_per_day):
    daily_averages = []
    current_sequence = np.copy(initial_sequence.reshape((1, -1, initial_sequence.shape[-1])))

    for day in range(total_steps // steps_per_day):
        daily_forecast = []

        for _ in range(steps_per_day):
            # Predict the next step based on the current sequence
            forecasted_step_scaled = model.predict(current_sequence)[0]
            # Update the current sequence with the scaled prediction for the next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = forecasted_step_scaled

            # Inverse transform the scaled prediction to its original scale
            forecasted_step = scaler.inverse_transform(forecasted_step_scaled.reshape(1, -1))[0]

            # Append the 'bt' value (assuming it is the fourth feature) to daily_forecast
            daily_forecast.append(forecasted_step[3])

        # Calculate the daily average 'bt' value
        daily_average_bt = np.mean(daily_forecast)
        daily_averages.append(daily_average_bt)

    return np.array(daily_averages)
