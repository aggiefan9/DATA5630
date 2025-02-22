# -*- coding: utf-8 -*-
"""TS_app.py"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from xgboost import XGBRegressor
import xgboost as xgb


def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def get_x_y(df, target, train_size, num_lags):
    data = df.copy()
    series = data[target].dropna().to_numpy()
    lag_names = []
    for x in range(num_lags):
        lag = np.roll(series, 1)
        data[f'lag-{x+1}'] = lag
        lag_names.append(f'lag-{x+1}')

    X = np.array(data[lag_names])
    Y = data[[target]]
    split_point = int(len(data) * train_size)
    x_train, y_train = X[num_lags:split_point], Y[num_lags:split_point]
    x_test, y_test = X[split_point:], Y[split_point:]
    
    return x_train, y_train, x_test, y_test

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, mape


def run_forecast(x_train, y_train, x_test, y_test, model, fh, **kwargs):
    if model == 'ETS':
        forecaster = AutoETS(**kwargs)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        future_dates = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
        return forecaster, y_pred, y_forecast

    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        future_dates = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
        return forecaster, y_pred, y_forecast

    elif model == 'SVR':
        forecaster = SVR(**kwargs)
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        X_test = np.arange(len(y_train), len(y_train) + len(y_test)).reshape(-1, 1)

        # Scale the data
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Train and predict
        forecaster.fit(X_train, y_train_scaled)
        y_pred_scaled = forecaster.predict(X_test)
        y_pred = pd.Series(scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel(), index=y_test.index)

        # Forecast into the future
        future_dates = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
        X_future = np.arange(len(y_train) + len(y_test), len(y_train) + len(y_test) + fh).reshape(-1, 1)
        y_forecast_scaled = forecaster.predict(X_future)
        y_forecast = pd.Series(scaler.inverse_transform(y_forecast_scaled.reshape(-1, 1)).ravel(), index=future_dates)
        return forecaster, y_pred, y_forecast

    elif model == 'Random Forest':
        forecaster = RandomForestRegressor(**kwargs)
        forecaster.fit(x_train, y_train)
        y_pred_array = forecaster.predict(x_test)

        y_forecast_array = []

        input_X = x_test[-1]

        while len(y_forecast_array) < fh:
            prediction = forecaster.predict(input_X.reshape(1, -1))[0]
            y_forecast_array.append(prediction)

            input_X = np.roll(input_X, -1)
            input_X[-1] = prediction

        y_pred = pd.Series(y_pred_array, index=y_test.index)
        forecast_index = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
        y_forecast = pd.Series(y_forecast_array, index=forecast_index)
        return forecaster, y_pred, y_forecast   

    elif model == 'XGBoost':
        forecaster = XGBRegressor(**kwargs)
        forecaster.fit(x_train, y_train)
        y_pred_array = forecaster.predict(x_test)

        y_forecast_array = []

        input_X = x_test[-1]

        while len(y_forecast_array) < fh:
            prediction = forecaster.predict(input_X.reshape(1, -1))[0]
            y_forecast_array.append(prediction)

            input_X = np.roll(input_X, -1)
            input_X[-1] = prediction

        y_pred = pd.Series(y_pred_array, index=y_test.index)
        forecast_index = pd.period_range(start=y_test.index[-1] + 1, periods=fh, freq=y_train.index.freq)
        y_forecast = pd.Series(y_forecast_array, index=forecast_index)
        return forecaster, y_pred, y_forecast
    
    else:
        raise ValueError("Unsupported model")


def plot_time_series(y_train, y_test, y_pred, y_forecast, zoom_start, zoom_end, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train", alpha=0.5)
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    ax.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
    ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xlim(pd.Timestamp(f"{zoom_start}-01-01"), pd.Timestamp(f"{zoom_end}-12-31"))
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting App")

    col1, col2, col3 = st.columns([1.5, 3.5, 5])

    with col1:
        st.header("Model Assumptions")
        model_choice = st.selectbox("Select a model", ["ETS", "ARIMA", "Random Forest", "XGBoost", "SVR"])
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
            start_p = st.number_input("Min p", min_value=0, value=0)
            max_p = st.number_input("Max p", min_value=0, value=5)
            start_q = st.number_input("Min q", min_value=0, value=0)
            max_q = st.number_input("Max q", min_value=0, value=5)
            d = st.number_input("d", min_value=0, value=1)
            seasonal = st.checkbox("Seasonal", value=True)
            model_params = {"start_p": start_p, "max_p": max_p, "start_q": start_q, "max_q": max_q, "d": d, "seasonal": seasonal}
        
        elif model_choice == "SVR":
            st.markdown("### SVR Parameters")
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            C = st.number_input("C (Regularization)", min_value=0.01, value=1.0, step=0.01)
            epsilon = st.number_input("Epsilon (Tolerance)", min_value=0.0, value=0.1, step=0.01)
            model_params = {"kernel": kernel, "C": C, "epsilon": epsilon}

        elif model_choice == "Random Forest":
            st.subheader("Random Forest Hyperparameters")
            n_estimators = st.number_input("Number of Trees (n_estimators)", min_value=10, value=100, step=10)
            max_depth = st.number_input("Max Depth (None for unlimited)", min_value=1, value=None)
            random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42)
            num_lags = st.number_input("Lags", min_value=0, value=12)
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth else None,
                "random_state": random_state,
            }
        
        elif model_choice == "XGBoost":
            st.subheader("XGBoost Hyperparameters")
            n_estimators = st.number_input("Number of Trees (n_estimators)", min_value=10, value=100, step=10)
            max_depth = st.number_input("Max Depth (None for unlimited)", min_value=1, value=None)
            learning_rate = st.number_input("Learning Rate (eta)", min_value=0.01, value=0.1, step=0.01)
            subsample = st.number_input("Subsample (fraction of samples)", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42)
            num_lags = st.number_input("Lags", min_value=0, value=12)
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth if max_depth else None,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "random_state": random_state,
            }

    with col2:
        st.header("Data Handling")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                country_options = df['country'].unique()
                country_choice = st.selectbox("Select a country", country_options)
                df = df[df['country'] == country_choice]

                freq_options = ['M', 'Q', 'Y']
                freq = st.selectbox("Select the data frequency", freq_options)

                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                df = df.set_index('date')
                df = df.sort_index()  # Ensure the index is sorted
                df.index = df.index.to_period(freq)
                df = df.groupby([df.index, 'country']).sum().reset_index().set_index('date')

                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                target_variable = st.selectbox("Select your target variable", numeric_columns)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index.to_timestamp(), df[target_variable])
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    with col3:
        st.header("Forecast Results")
        fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
        run_forecast_button = st.button("Run Forecast")

        if run_forecast_button:
            if 'df' in locals() and 'target_variable' in locals():
                try:
                    y = df[target_variable]
                    y_train, y_test = manual_train_test_split(y, train_size)
                    if model_choice in ['Random Forest', 'XGBoost']:
                        x_train, y_train, x_test, y_test = get_x_y(df, target_variable, train_size, num_lags)
                        forecaster, y_pred, y_forecast = run_forecast(x_train, y_train, x_test, y_test, model_choice, fh, **model_params)
                    else:
                        forecaster, y_pred, y_forecast = run_forecast(None, y_train, None, y_test, model_choice, fh, **model_params)

                    # Full dataset view
                    fig_full, ax_full = plt.subplots(figsize=(12, 6))
                    ax_full.plot(y_train.index.to_timestamp(), y_train.values, label="Train", alpha=0.5)
                    ax_full.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
                    ax_full.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
                    ax_full.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
                    plt.legend()
                    plt.title(f"{model_choice} Forecast for {target_variable}")
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    st.pyplot(fig_full)

                    # Zoomed-in view near the forecast window
                    forecast_start = y_test.index[0].start_time
                    forecast_end = y_forecast.index[-1].start_time
                    extended_start = forecast_start - pd.DateOffset(months=6)  # Show 6 months before
                    extended_end = forecast_end + pd.DateOffset(months=6)    # Show 6 months after

                    fig_zoom, ax_zoom = plt.subplots(figsize=(12, 6))
                    ax_zoom.plot(y_train.index.to_timestamp(), y_train.values, label="Train", alpha=0.5)
                    ax_zoom.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
                    ax_zoom.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
                    ax_zoom.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
                    plt.legend()
                    plt.title(f"Zoomed-In {model_choice} Forecast for {target_variable}")
                    plt.xlabel("Date")
                    plt.ylabel("Value")
                    plt.xlim(extended_start, extended_end)  # Set zoomed x-axis range
                    st.pyplot(fig_zoom)

                    # Compute and display metrics
                    y_test = pd.Series(y_test.values.flatten())
                    y_pred = pd.Series(y_pred.values.flatten()) 
                    
                    # Remove NaN values
                    valid_indices = (~y_test.isna()) & (~y_pred.isna())
                    y_test = y_test[valid_indices]
                    y_pred = y_pred[valid_indices]

                    # Exclude zero values in y_test to avoid division by zero
                    nonzero_indices = y_test != 0
                    y_test = y_test[nonzero_indices]
                    y_pred = y_pred[nonzero_indices]
                    
                    mse, mae, mape = compute_metrics(y_test, y_pred)
                    
                    # Initialize a metrics DataFrame at the beginning of the script
                    if "metrics_table" not in st.session_state:
                        st.session_state["metrics_table"] = pd.DataFrame(columns=["Country", "Model", "MSE", "MAE", "MAPE (%)"])

                    # After computing the metrics
                    model_name = model_choice  # Use the selected model's name
                    country_name = country_choice  # Use the selected country's name

                    # Append the metrics to the table
                    new_metrics = {
                        "Country": country_name,  
                        "Model": model_name,
                        "MSE": mse,
                        "MAE": mae,
                        "MAPE (%)": mape
                    }
                    st.session_state["metrics_table"] = pd.concat(
                        [st.session_state["metrics_table"], pd.DataFrame([new_metrics])],
                        ignore_index=True
                    )

                    # Display the metrics table
                    st.subheader("Model Comparison Table")
                    st.dataframe(st.session_state["metrics_table"])

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please upload data and select a target variable first.")


if __name__ == "__main__":
    main()

