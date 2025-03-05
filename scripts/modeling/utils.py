import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ForecastPlotter:
    def __init__(self, actual, arima_pred, sarima_pred, lstm_pred, test_index, scaler):
        """
        Initialize the ForecastPlotter class.
        
        Parameters:
            actual (numpy array): Actual test set values.
            arima_pred (numpy array): ARIMA model predictions.
            sarima_pred (numpy array): SARIMA model predictions.
            lstm_pred (numpy array): LSTM model predictions.
            test_index (pandas datetime index): Date index for the test set.
            scaler (MinMaxScaler): Scaler used for inverse transformation.
        """
        self.actual = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        self.arima_pred = scaler.inverse_transform(np.array(arima_pred).reshape(-1, 1)).flatten()
        self.sarima_pred = scaler.inverse_transform(np.array(sarima_pred).reshape(-1, 1)).flatten()
        self.lstm_pred = scaler.inverse_transform(np.array(lstm_pred).reshape(-1, 1)).flatten()
        self.test_index = test_index

    def plot_forecasts(self, title="TSLA Stock Price Forecasting"):
        """
        Plots actual vs predicted stock prices for ARIMA, SARIMA, and LSTM.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.test_index, self.actual, label="Actual Prices", color="black")
        plt.plot(self.test_index, self.arima_pred, label="ARIMA Predictions", linestyle="dashed", color="red")
        plt.plot(self.test_index, self.sarima_pred, label="SARIMA Predictions", linestyle="dashed", color="blue")
        plt.plot(self.test_index, self.lstm_pred, label="LSTM Predictions", linestyle="dashed", color="green")

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

