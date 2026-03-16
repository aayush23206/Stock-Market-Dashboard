import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches OHLCV data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_info(ticker: str) -> dict:
    """Returns name, sector, market cap, PE ratio, 52W high/low, current price."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Safely extract values with fallbacks
        return {
            "name": info.get("shortName", info.get("longName", ticker)),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", info.get("forwardPE", "N/A")),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
        }
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return {}

def get_multiple_stocks(tickers: list, period: str) -> pd.DataFrame:
    """Fetches data for multiple tickers to compare."""
    try:
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty and 'Close' in hist.columns:
                data[ticker] = hist['Close']
        
        if data:
            df = pd.DataFrame(data)
            # Forward fill and backward fill to handle missing values from different market holidays
            df = df.ffill().bfill()
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching multiple stocks: {e}")
        return pd.DataFrame()
