import pandas as pd
import os

class FinanceDataCleaner:
    """
    A class to clean financial data.
    """
    
    def __init__(self, data_path="../resources/data/"):
        self.data_path = data_path
    
    def load_data(self):
        """
        Load data from CSV files.
        Returns a dictionary of cleaned DataFrames.
        """
        files = os.listdir(self.data_path)
        data = {}
        for file in files:
            if file.endswith(".csv"):
                ticker = file.split(".")[0]
                df = pd.read_csv(f"{self.data_path}{file}", parse_dates=["Date"], index_col="Date")
                data[ticker] = df
        return data
    
    def check_missing_values(self, data_dict):
        """
        Check for missing values in datasets.
        """
        for ticker, df in data_dict.items():
            missing = df.isnull().sum()
            print(f"\nMissing values in {ticker}:")
            print(missing[missing > 0])

    def handle_missing_values(self, data_dict, method="ffill"):
        """
        Handle missing values in financial datasets.
        Method options: 'ffill' (forward fill), 'bfill' (backward fill), 'drop' (remove rows).
        """
        for ticker, df in data_dict.items():
            if method == "ffill":
                df.fillna(method="ffill", inplace=True)
            elif method == "bfill":
                df.fillna(method="bfill", inplace=True)
            elif method == "drop":
                df.dropna(inplace=True)
            data_dict[ticker] = df
        return data_dict
