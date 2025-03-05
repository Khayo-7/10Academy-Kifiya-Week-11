import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class TimeSeriesPreprocessor:
    """
    Prepares time series data for forecasting.
    """

    def __init__(self, df, target_col="Close"):
        self.df = df[[target_col]].copy()
        self.target_col = target_col
        self.scaler = MinMaxScaler()

    def test_stationarity(self):
        """
        Perform Augmented Dickey-Fuller (ADF) test to check stationarity.
        """
        result = adfuller(self.df[self.target_col].dropna())
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        if result[1] < 0.05:
            print("The series is stationary.")
        else:
            print("The series is non-stationary. Differencing might be needed.")

    def make_stationary(self):
        """
        Apply differencing if data is non-stationary.
        """
        self.df["Differenced"] = self.df[self.target_col].diff().dropna()
        return self.df

    def scale_data(self):
        """
        Scale data for LSTM.
        """
        self.df[self.target_col] = self.scaler.fit_transform(self.df[self.target_col].values.reshape(-1, 1))
        return self.df, self.scaler

    def split_data(self, train_size=0.8):
        """
        Split data into train and test sets.
        """
        train_size = int(len(self.df) * train_size)
        train, test = self.df.iloc[:train_size], self.df.iloc[train_size:]
        return train, test

