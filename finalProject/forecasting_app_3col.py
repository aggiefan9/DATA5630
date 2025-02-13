# Author: Prof. Pedram Jahangiry
# Date: 2024-10-10

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.utils import timeseries_dataset_from_array


def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def rnn_train_test_split(y, train_size, sequence_length=60, epochs=0, hidden_units=0):
    sequence_length = sequence_length
    h = 1
    delay = sequence_length  + h - 1
    batch_size = 32

    train_dataset = timeseries_dataset_from_array(
        data = y[:-delay],
        targets=y[delay:],
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=0,
        end_index=train_size)

    test_dataset = timeseries_dataset_from_array(
        data = y[:-delay],
        targets=y[delay:],
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=train_size)
    
    return train_dataset, test_dataset

def run_forecast(y_train, y_test, model, fh, **kwargs):
    if model == 'ETS':
        forecaster = AutoETS(**kwargs)
    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)
    else:
        raise ValueError("Unsupported model")
    
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
    
    last_date = y_test.index[-1]
    future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
    future_horizon = ForecastingHorizon(future_dates, is_relative=False)
    y_forecast = forecaster.predict(fh=future_horizon)
    
    return forecaster, y_pred, y_forecast

def run_rnn_forecast(y_train, y_test, fh, epochs=50, sequence_length=30, hidden_units=32):

    inputs = Input(shape=(sequence_length, 1))
    x = SimpleRNN(hidden_units, return_sequences=False)(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    
    model.fit(y_train, epochs=epochs, verbose=0)
    
    y_pred = model.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    last_date = y_test.index[-1]
    future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
    future_horizon = ForecastingHorizon(future_dates, is_relative=False)
    y_forecast = model.predict(fh=future_horizon)
    
    return y_pred, y_forecast


def plot_time_series(y_train, y_test, y_pred, y_forecast, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    ax.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
    ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig



def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    col1, col2, col3 = st.columns([1.5, 3.5, 5])

    with col1:
        st.header("Model Assumptions")
        model_choice = st.selectbox("Select a model", ["ETS", "ARIMA", "RNN", "Random Forest"])
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100

        if model_choice == "ETS":
            error = st.selectbox("Error type", ["add", "mul"])
            trend = st.selectbox("Trend type", ["add", "mul", None])
            seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
            damped_trend = st.checkbox("Damped trend", value=False)
            seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=1)
            model_params = {
                "error": error,
                "trend": trend,
                "seasonal": seasonal,
                "damped_trend": damped_trend,
                "sp": seasonal_periods,
            }
        elif model_choice == "ARIMA":
            st.subheader("Non-seasonal")
            start_p = st.number_input("Min p", min_value=0, value=0)
            max_p = st.number_input("Max p", min_value=0, value=5)
            start_q = st.number_input("Min q", min_value=0, value=0)
            max_q = st.number_input("Max q", min_value=0, value=5)
            d = st.number_input("d", min_value=0, value=1)
            
            st.subheader("Seasonal")
            seasonal = st.checkbox("Seasonal", value=True)
            if seasonal:
                start_P = st.number_input("Min P", min_value=0, value=0)
                max_P = st.number_input("Max P", min_value=0, value=2)
                start_Q = st.number_input("Min Q", min_value=0, value=0)
                max_Q = st.number_input("Max Q", min_value=0, value=2)
                D = st.number_input("D", min_value=0, value=1)
                sp = st.number_input("Periods", min_value=1, value=12)
            
            model_params = {
                "start_p": start_p,
                "max_p": max_p,
                "start_q": start_q,
                "max_q": max_q,
                "d": d,
                "seasonal": seasonal,
            }
            if seasonal:
                model_params.update({
                    "start_P": start_P,
                    "max_P": max_P,
                    "start_Q": start_Q,
                    "max_Q": max_Q,
                    "D": D,
                    "sp": sp
                })
        elif model_choice == "RNN":
            st.subheader("RNN Hyperparameters")
            epochs = st.number_input("Epochs", min_value=1, value=50)
            sequence_length = st.number_input("Sequence Length", min_value=1, value=30)
            hidden_units = st.number_input("Hidden Layer Units", min_value=1, value=16)
            # hidden_units = st.text_input("Hidden Layer Units (comma-separated)", value="64,32")
            # hidden_units = [int(x.strip()) for x in hidden_units.split(",") if x.strip().isdigit()]
            model_params = {
                "epochs": epochs,
                "sequence_length": sequence_length,
                "hidden_units": hidden_units,
            }
        elif model_choice == "Random Forest":
            st.subheader("Random Forest Hyperparameters")
            n_estimators = st.number_input("Number of Trees (n_estimators)", min_value=10, value=100, step=10)
            max_depth = st.number_input("Max Depth (None for unlimited)", min_value=1, value=None)
            random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42)
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth else None,
                "random_state": random_state,
            }

    with col2:
        st.header("Data Handling")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Allow user to select the frequency
                freq_options = ['D', 'W', 'M', 'Q', 'Y']
                freq = st.selectbox("Select the data frequency", freq_options)
                
                # Convert the index to datetime and then to PeriodIndex
                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                df = df.set_index('date')
                df = df.sort_index()  # Ensure the index is sorted
                df.index = df.index.to_period(freq)
                
                # Remove any rows with NaT in the index
                df = df.loc[df.index.notnull()]
                
                st.subheader("Data Preview")
                st.write(df.head())

                # Filter out non-numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("No numeric columns found in the uploaded data. Please ensure your CSV contains numeric data for forecasting.")
                else:
                    target_variable = st.selectbox("Select your target variable", numeric_columns)

                    # Plot the time series of the selected target variable
                    st.subheader(f"Time Series Plot: {target_variable}")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df.index.to_timestamp(), df[target_variable])
                    plt.title(f"{target_variable} Time Series")
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
                st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")

    with col3:
        st.header("Forecast Results")
        fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
        run_forecast_button = st.button("Run Forecast")
        
        if run_forecast_button:
            if 'df' in locals() and 'target_variable' in locals():
                try:
                    y = df[target_variable]

                    if model_choice == "RNN":
                        y_train, y_test = rnn_train_test_split(y, train_size, **model_params)
                        y_pred, y_forecast = run_rnn_forecast(y_train, y_test, fh, **model_params)
                    else:
                        y_train, y_test = manual_train_test_split(y, train_size)
                        forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model_choice, fh, **model_params)

                    fig = plot_time_series(y_train, y_test, y_pred, y_forecast, f"{model_choice} Forecast for {target_variable}")
                    st.pyplot(fig)

                    st.subheader("Test Set Predictions")
                    st.write(y_pred)

                    st.subheader("Future Forecast Values")
                    st.write(y_forecast)
                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable before running the forecast.")

if __name__ == "__main__":
    main()