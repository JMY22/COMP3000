from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Bidirectional, Conv1D
from keras.src.metrics import MeanSquaredError
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
from tensorflow.keras.regularizers import l2


def build_model(n_features, n_steps, output_units=32, dropout_rate=0.3, regularization_rate=0.0005):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Bidirectional(LSTM(output_units, return_sequences=True, activation='tanh', kernel_regularizer=l2(regularization_rate))),
        Dropout(dropout_rate),
        LSTM(output_units, activation='tanh', kernel_regularizer=l2(regularization_rate)),
        Dropout(dropout_rate),
        Dense(n_features, activation='relu', kernel_regularizer=l2(regularization_rate))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_and_save_model(X_train, y_train, X_val, y_val, model_path, n_features, n_steps, epochs=50, batch_size=128):
    model = build_model(n_features, n_steps)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
              callbacks=[early_stopping, reduce_lr])
    model.save(model_path)
    return model


def load_and_update_model(model_path, X_train, y_train, X_val, y_val, epochs=5, batch_size=128):
    try:
        model = load_model(model_path, custom_objects={"Adam": Adam, "MeanSquaredError": MeanSquaredError})
        print("Model loaded successfully!")
    except ValueError as e:
        print(f"Error loading model. Rebuilding model. Error: {e}")
        model = build_model(n_features=X_train.shape[-1], n_steps=X_train.shape[1], output_units=32, dropout_rate=0.3)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        print("New model initialized due to load failure.")

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001)])
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
