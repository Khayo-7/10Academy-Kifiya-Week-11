import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ARIMAModel:
    """
    Implements ARIMA for time series forecasting.
    """

    def __init__(self, train, test, order=(5,1,0)):
        self.train = train
        self.test = test
        self.order = order
        self.model = None

    def train_model(self):
        """
        Train ARIMA model.
        """
        self.model = ARIMA(self.train, order=self.order).fit()
        print(self.model.summary())

    def forecast(self, steps=30):
        """
        Forecast future values.
        """
        forecast = self.model.forecast(steps=steps)
        return forecast

    def evaluate(self):
        """
        Compute RMSE and MAE.
        """
        predictions = self.model.forecast(steps=len(self.test))
        rmse = np.sqrt(mean_squared_error(self.test, predictions))
        mae = mean_absolute_error(self.test, predictions)
        print(f"ARIMA RMSE: {rmse}, MAE: {mae}")
        return predictions

