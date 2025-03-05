import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

class FinanceEDA:
    """
    A class to perform exploratory data analysis (EDA) on financial datasets.
    """
    
    def __init__(self, data):
        self.data = data
    
    def plot_closing_prices(self):
        """
        Plot the closing prices of all assets.
        """
        plt.figure(figsize=(12,6))
        for ticker, df in self.data.items():
            plt.plot(df.index, df["Close"], label=ticker)
        plt.legend()
        plt.title("Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid()
        plt.show()
    
    def calculate_daily_returns(self):
        """
        Compute daily returns for all assets.
        """
        returns = {}
        for ticker, df in self.data.items():
            df["Daily Return"] = df["Close"].pct_change()
            returns[ticker] = df["Daily Return"]
        return returns

    def calculate_returns(self):
        return self.data.pct_change().dropna()

    def detect_outliers(self):
        outliers = {}
        # returns = self.calculate_returns()
        returns = self.calculate_daily_returns()
        for ticker, ret in returns.items():
            z_scores = (ret - ret.mean()) / ret.std()
            outliers[ticker] = ret[abs(z_scores) > 3]
        return outliers

    def get_sharpe_ratios(self):
        sharpe_ratios = {}
        # returns = self.calculate_returns()
        returns = self.calculate_daily_returns()
        for ticker, ret in returns.items():
            sharpe_ratios[ticker] = ret.mean() / ret.std() * np.sqrt(252)

        return sharpe_ratios

    def get_VaRs(self, percentile=95):
        VaRs = {}
        # returns = self.calculate_returns()
        returns = self.calculate_daily_returns()
        for ticker, ret in returns.items():
            VaRs[ticker] = np.percentile(ret.dropna(), 100 - percentile) * 100
        return VaRs

    def plot_volatility(self, window=30):
        """
        Plot rolling standard deviation for volatility analysis.
        """
        plt.figure(figsize=(12,6))
        for ticker, df in self.data.items():
            rolling_vol = df["Close"].pct_change().rolling(window).std()
            plt.plot(rolling_vol, label=f"{ticker} {window}-day Volatility")
        plt.legend()
        plt.title(f"Rolling {window}-Day Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.grid()
        plt.show()
    
    def decompose_time_series(self, ticker):
        """
        Decompose time series into trend, seasonality, and residual components.
        """
        df = self.data[ticker]
        decomposition = seasonal_decompose(df["Close"], model='additive', period=252)
        fig, axes = plt.subplots(4, 1, figsize=(12,8))
        decomposition.observed.plot(ax=axes[0], title="Observed")
        decomposition.trend.plot(ax=axes[1], title="Trend")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
        decomposition.resid.plot(ax=axes[3], title="Residual")
        plt.tight_layout()
        plt.show()
