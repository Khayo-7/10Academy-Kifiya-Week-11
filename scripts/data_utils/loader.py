import yfinance as yf
import pandas as pd

class FinanceDataLoader:
    """
    A class to fetch historical stock data from Yahoo Finance.
    """
    
    def __init__(self, tickers, start_date="2015-01-01", end_date="2025-01-31"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
    
    def fetch_data(self):
        """
        Fetch historical financial data for given tickers.
        Returns a dictionary of DataFrames.
        """
        data = {}
        for ticker in self.tickers:
            stock = yf.download(ticker, start=self.start_date, end=self.end_date)
            stock["Ticker"] = ticker
            data[ticker] = stock
        # data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close'] # ['Adj Close']
        # data.dropna(inplace=True)
        return data
    
    def save_data(self, data_dict, save_path="../resources/data/"):
        """
        Save data to CSV files.
        """
        for ticker, df in data_dict.items():
            df.to_csv(f"{save_path}{ticker}.csv")
        print(f"Data saved in {save_path}")
