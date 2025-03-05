import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, df, risk_free_rate=0.01):
        """
        Initialize with a DataFrame containing daily closing prices for assets.
        Columns should include: 'TSLA', 'BND', 'SPY'.
        
        Parameters:
            df (pd.DataFrame): DataFrame with asset prices indexed by Date.
            risk_free_rate (float): Annual risk-free rate used for Sharpe Ratio calculation.
        """
        self.df = df.copy()
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.annual_returns = None
        self.cov_matrix = None
        self.weights_opt = None

    def compute_returns(self):
        """Compute daily percentage returns and annualized returns."""
        self.returns = self.df.pct_change().dropna()
        # Assuming ~252 trading days per year
        self.annual_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        return self.returns, self.annual_returns, self.cov_matrix

    def portfolio_performance(self, weights):
        """
        Calculate expected portfolio return, volatility, and Sharpe ratio.
        
        Parameters:
            weights (np.array): Portfolio weights for each asset.
        
        Returns:
            tuple: (expected return, volatility, Sharpe ratio)
        """
        port_return = np.dot(self.annual_returns, weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
        return port_return, port_volatility, sharpe_ratio

    def negative_sharpe(self, weights):
        """Objective function to minimize (negative Sharpe Ratio)."""
        _, _, sharpe = self.portfolio_performance(weights)
        return -sharpe

    def optimize_portfolio(self):
        """
        Optimize portfolio weights to maximize the Sharpe Ratio.
        
        Returns:
            np.array: Optimal portfolio weights.
        """
        num_assets = len(self.df.columns)
        # Initial guess: equal weight distribution.
        init_guess = np.repeat(1/num_assets, num_assets)
        # Constraint: sum of weights must equal 1.
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: weights between 0 and 1 (long-only portfolio)
        bounds = tuple((0, 1) for asset in range(num_assets))
        
        optimal = minimize(self.negative_sharpe, init_guess, method='SLSQP',
                           bounds=bounds, constraints=constraints)
        self.weights_opt = optimal.x
        return self.weights_opt

    def compute_portfolio_returns(self, weights=None):
        """
        Compute daily portfolio returns given asset weights.
        
        Parameters:
            weights (np.array): If not provided, use optimized weights.
        
        Returns:
            pd.Series: Daily portfolio returns.
        """
        if weights is None:
            if self.weights_opt is None:
                raise ValueError("Please run optimize_portfolio() first or provide weights.")
            weights = self.weights_opt
        port_returns = self.returns.dot(weights)
        return port_returns

    def plot_portfolio_performance(self, weights=None):
        """
        Plot cumulative returns and risk-return scatter plot.
        
        Parameters:
            weights (np.array): Portfolio weights. If not provided, uses optimized weights.
        """
        if weights is None:
            if self.weights_opt is None:
                raise ValueError("Please run optimize_portfolio() first or provide weights.")
            weights = self.weights_opt

        # Compute portfolio returns and cumulative returns
        port_returns = self.compute_portfolio_returns(weights)
        cumulative_returns = (1 + port_returns).cumprod()

        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, label="Optimized Portfolio")
        plt.title("Cumulative Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Compute individual asset performance for risk-return comparison
        asset_returns = self.annual_returns
        asset_volatility = np.sqrt(np.diag(self.cov_matrix))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(asset_volatility, asset_returns, c='blue', label="Individual Assets")
        for i, txt in enumerate(self.df.columns):
            plt.annotate(txt, (asset_volatility[i], asset_returns[i]), fontsize=12)
        
        # Plot portfolio risk and return
        port_return, port_volatility, sharpe = self.portfolio_performance(weights)
        plt.scatter(port_volatility, port_return, c='red', label="Optimized Portfolio", marker='X', s=200)
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Return")
        plt.title("Risk-Return Profile")
        plt.legend()
        plt.grid(True)
        plt.show()

    def display_optimal_weights(self):
        """
        Display optimal weights for each asset.
        """
        if self.weights_opt is None:
            raise ValueError("Please run optimize_portfolio() first.")
        weights_df = pd.DataFrame({
            "Asset": self.df.columns,
            "Optimal Weight": self.weights_opt
        })
        print(weights_df)
