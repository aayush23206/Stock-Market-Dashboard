"""
Exploratory Data Analysis module for stock market data.
Performs correlation analysis, descriptive statistics, and data profiling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class StockEDA:
    """Exploratory Data Analysis for stock market data."""
    
    def __init__(self, df: pd.DataFrame, ticker: str = None):
        """
        Initialize EDA analyzer.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
        """
        self.df = df.copy()
        self.ticker = ticker
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate derived metrics."""
        if 'Close' in self.df.columns:
            self.df['Returns'] = self.df['Close'].pct_change()
            self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
            self.df['MA_20'] = self.df['Close'].rolling(window=20).mean()
            self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
            self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
            
            # Bollinger Bands
            self.df['BB_Middle'] = self.df['Close'].rolling(window=20).mean()
            self.df['BB_Std'] = self.df['Close'].rolling(window=20).std()
            self.df['BB_Upper'] = self.df['BB_Middle'] + (self.df['BB_Std'] * 2)
            self.df['BB_Lower'] = self.df['BB_Middle'] - (self.df['BB_Std'] * 2)
            
            # RSI (Relative Strength Index)
            self.df['RSI'] = self._calculate_rsi(self.df['Close'])
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if 'Close' not in self.df.columns:
            return {}
        
        close_prices = self.df['Close']
        returns = self.df['Returns'].dropna()
        
        return {
            'current_price': close_prices.iloc[-1],
            'avg_price': close_prices.mean(),
            'min_price': close_prices.min(),
            'max_price': close_prices.max(),
            'std_dev': close_prices.std(),
            'avg_daily_return': returns.mean(),
            'daily_volatility': returns.std(),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'total_return': (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0],
            'max_drawdown': self._calculate_max_drawdown(close_prices),
        }
    
    @staticmethod
    def _calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        excess_return = returns.mean() - (risk_free_rate / 252)
        return (excess_return * 252) / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def _calculate_max_drawdown(prices):
        """Calculate maximum drawdown."""
        cumulative = (1 + (prices.pct_change())).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class CorrelationAnalyzer:
    """Analyze correlations between multiple stocks."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize correlation analyzer.
        
        Args:
            df: DataFrame with 'Close' prices for multiple tickers (columns are tickers)
        """
        self.df = df.copy()
        self.correlation_matrix = df.corr()
        self.returns = df.pct_change().dropna()
        self.returns_correlation = self.returns.corr()
    
    def get_correlation_insights(self, ticker: str) -> Dict:
        """Get correlation insights for a specific ticker."""
        if ticker not in self.correlation_matrix.columns:
            return {}
        
        correlations = self.correlation_matrix[ticker].sort_values(ascending=False)
        returns_corr = self.returns_correlation[ticker].sort_values(ascending=False)
        
        return {
            'price_correlations': correlations.to_dict(),
            'returns_correlations': returns_corr.to_dict(),
            'most_correlated': correlations.index[1] if len(correlations) > 1 else None,
            'most_correlated_value': correlations.iloc[1] if len(correlations) > 1 else None,
            'least_correlated': correlations.index[-1],
            'least_correlated_value': correlations.iloc[-1],
        }
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get price correlation matrix."""
        return self.correlation_matrix
    
    def get_returns_correlation_matrix(self) -> pd.DataFrame:
        """Get returns correlation matrix."""
        return self.returns_correlation
    
    def find_diversification_pairs(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """Find stock pairs with low correlation (good for diversification)."""
        pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                ticker1 = self.correlation_matrix.columns[i]
                ticker2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                if abs(corr_value) < threshold:
                    pairs.append((ticker1, ticker2, corr_value))
        
        return sorted(pairs, key=lambda x: abs(x[2]))


def prepare_data_for_forecasting(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Prepare data for forecasting.
    
    Args:
        df: Historical price data with 'Close' column
        test_size: Fraction of data for testing
    
    Returns:
        Tuple of (train_data, test_data, full_series)
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    close_prices = df['Close'].dropna()
    split_idx = int(len(close_prices) * (1 - test_size))
    
    train_data = close_prices[:split_idx]
    test_data = close_prices[split_idx:]
    
    return train_data, test_data, close_prices
