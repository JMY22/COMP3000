from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np


def build_model(n_features, n_steps, output_units=25, dropout_rate=0.5):
    model = Sequential([
        Bidirectional(LSTM(output_units, return_sequences=True, activation='tanh'), input_shape=(n_steps, n_features)),
        BatchNormalization(),
        Dropout(dropout_rate),
        LSTM(output_units // 2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(n_features)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    return model


def train_and_save_model(X_train, y_train, X_val, y_val, model_path, n_features, n_steps, epochs=50, batch_size=128):
    model = build_model(n_features, n_steps)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr])
    model.save(model_path)
    return model


def load_and_update_model(model_path, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    model = load_model(model_path)
    # Assuming validation or rebuilding based on n_features and n_steps is not needed here
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr])
    model.save(model_path)
    return model


def forecast(model, scaler, initial_sequence, steps=100, noise_level=0.01):
    n_features = 4  # Defined as per your constants
    initial_sequence = initial_sequence.reshape((1, -1, n_features))  # Ensure correct shape

    forecasts = np.zeros((steps, n_features))
    current_sequence = initial_sequence

    for i in range(steps):
        noisy_sequence = current_sequence + np.random.normal(0, noise_level, current_sequence.shape)
        forecasted_step = model.predict(noisy_sequence)[0]
        forecasts[i] = forecasted_step
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = forecasted_step

    forecasts = scaler.inverse_transform(forecasts)
    return forecasts


def forecast_average_bt(model, scaler, initial_sequence, total_steps, steps_per_day, n_steps, noise_level=0.15):
    n_features = 4  # Defined as per your constants
    initial_sequence = initial_sequence.reshape((1, n_steps, n_features))  # Ensure correct shape for n_steps

    daily_averages = []
    current_sequence = initial_sequence

    for day in range(total_steps // steps_per_day):
        daily_forecast = []
        for _ in range(steps_per_day):
            noisy_sequence = current_sequence + np.random.normal(0, noise_level, current_sequence.shape)
            forecasted_step_scaled = model.predict(noisy_sequence)[0]
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = forecasted_step_scaled
            forecasted_step = scaler.inverse_transform(forecasted_step_scaled.reshape(1, -1))[0]
            daily_forecast.append(forecasted_step[3])

        daily_average_bt = np.mean(daily_forecast)
        daily_averages.append(daily_average_bt)

    return np.array(daily_averages)
