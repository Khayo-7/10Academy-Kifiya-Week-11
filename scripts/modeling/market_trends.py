import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MarketTrendForecaster:
    """
    A class to forecast future market trends using a trained time series model.
    Supports ARIMA, SARIMA, and a naive multi-step approach for LSTM.
    """
    def __init__(self, model, model_name, last_train_date, scaler=None):
        """
        Parameters:
            model: The trained model instance.
                   For ARIMA/SARIMA, this is the fitted statsmodels model (accessible via model.model).
                   For LSTM, this is the instance of LSTMModel (which must have train, lookback, and forecast() method).
            model_name (str): The type of model ("ARIMA", "SARIMA", or "LSTM").
            last_train_date (pd.Timestamp): The last date from the training set.
            scaler: A scaler used for inverse transforming data (optional).
        """
        self.model = model
        self.model_name = model_name.lower()
        self.last_train_date = last_train_date
        self.scaler = scaler

    def forecast_future(self, forecast_steps=126):
        """
        Forecast future values for the specified number of steps.
        For ARIMA/SARIMA, returns predicted means and confidence intervals.
        For LSTM, uses a naive iterative forecasting method (confidence intervals are not available).

        Parameters:
            forecast_steps (int): Number of days to forecast (e.g., 126 for ~6 months).

        Returns:
            forecast (pd.Series or np.array): The forecasted values.
            conf_int (pd.DataFrame or None): Confidence intervals (if available).
        """
        if self.model_name in ["arima", "sarima"]:
            # Use statsmodels' get_forecast to generate forecast and confidence intervals.
            forecast_obj = self.model.model.get_forecast(steps=forecast_steps)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            # Inverse transform if a scaler is provided.
            if self.scaler:
                forecast = self.scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
                lower = self.scaler.inverse_transform(conf_int.iloc[:, 0].values.reshape(-1, 1)).flatten()
                upper = self.scaler.inverse_transform(conf_int.iloc[:, 1].values.reshape(-1, 1)).flatten()
                conf_int = pd.DataFrame({"lower": lower, "upper": upper})
            else:
                forecast = forecast.values
            return forecast, conf_int

        elif self.model_name == "lstm":
            # Naively forecast future values using iterative predictions.
            predictions = []
            # For LSTM, assume self.model.train is a numpy array of training data.
            last_window = self.model.train[-self.model.lookback:]
            for i in range(forecast_steps):
                pred = self.model.forecast(last_window)
                predictions.append(pred)
                # Update the window: remove the first element and append the new prediction.
                last_window = np.append(last_window[1:], pred)
            predictions = np.array(predictions)
            if self.scaler:
                predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            # Confidence intervals are not available for LSTM in this simple approach.
            return predictions, None
        else:
            raise ValueError("Unsupported model type. Please choose ARIMA, SARIMA, or LSTM.")

    def plot_forecast(self, historical_data, forecast, conf_int=None, title="Future Market Trends Forecast"):
        """
        Plots historical data along with the forecast and confidence intervals.

        Parameters:
            historical_data (pd.Series): The historical stock prices (in original scale).
            forecast (np.array): Forecasted values (in original scale).
            conf_int (pd.DataFrame or None): Confidence intervals with 'lower' and 'upper' columns.
            title (str): Plot title.
        """
        # Generate date range for forecast period (assume business days).
        last_date = historical_data.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=len(forecast)+1, closed='right', freq='B')
        
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data.index, historical_data.values, label="Historical Data", color="black")
        plt.plot(forecast_index, forecast, label="Forecast", color="red", linestyle="--")
        if conf_int is not None:
            plt.fill_between(forecast_index, conf_int["lower"], conf_int["upper"], color="pink", alpha=0.3, label="Confidence Interval")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plt.show()
