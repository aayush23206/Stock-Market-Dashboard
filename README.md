# 📊 Live Stock Market Dashboard with EDA & Forecasting

A comprehensive, production-ready financial analytics platform combining real-time market data, exploratory data analysis (EDA), correlation studies, and advanced time-series forecasting models.

## 🎯 Features

### 📈 **Dashboard (app.py)**
- Real-time stock price data via yfinance
- Interactive candlestick and line charts
- Technical indicators: SMA, EMA, Bollinger Bands
- Multi-stock comparison
- Key metrics: Market Cap, PE Ratio, 52W High/Low
- Full-featured Streamlit interface

### 📊 **Exploratory Data Analysis (eda_analysis.py)**
- **Statistical Summary**
  - Price statistics (min, max, mean, std dev)
  - Daily returns analysis
  - Volatility metrics
  - Sharpe ratio calculation
  - Maximum drawdown analysis
  
- **Technical Indicators**
  - Moving Averages (20-day, 50-day)
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - Daily returns distribution

### 🔗 **Correlation Analysis (forecasting.py & eda_analysis.py)**
- Price correlation matrix across multiple stocks
- Daily returns correlation analysis
- Diversification opportunity identification
- Low-correlation stock pair detection
- Perfect for portfolio construction

### 🔮 **Time-Series Forecasting**

#### **ARIMA Model**
- Automatic order detection (p, d, q)
- ADF test for stationarity
- Confidence intervals (95%)
- Best for: Stationary/differenced time series
- Fast execution, interpretable results

#### **LSTM Neural Network**
- Multi-layer architecture with dropout
- Captures complex temporal dependencies
- Customizable lookback window
- Best for: Non-linear patterns, long-term trends
- Deep learning approach

#### **Model Comparison**
- Evaluate multiple models simultaneously
- Performance metrics: RMSE, MAE, MAPE, R²
- Side-by-side comparison
- Automated best model recommendation

## 📋 Project Structure

```
stock-dashboard/
├── app.py                 # Main dashboard interface
├── forecasting_app.py     # EDA & forecasting interface
├── eda_analysis.py        # EDA & correlation analysis
├── forecasting.py         # ARIMA & LSTM implementations
├── utils.py               # Data fetching utilities
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Pip or Conda

### 1. Clone/Setup Project
```bash
cd stock-dashboard
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application

**Main Dashboard:**
```bash
streamlit run app.py
```

**EDA & Forecasting Analytics:**
```bash
streamlit run forecasting_app.py
```

Application will open at `http://localhost:8501`

## 📚 Module Documentation

### `eda_analysis.py`

#### `StockEDA` Class
```python
from eda_analysis import StockEDA, prepare_data_for_forecasting

# Initialize analyzer
df = get_stock_data("AAPL", "2y", "1d")
eda = StockEDA(df, ticker="AAPL")

# Get summary statistics
stats = eda.get_summary_stats()
# Returns: current_price, avg_price, min_price, max_price, 
#          std_dev, avg_daily_return, daily_volatility, 
#          sharpe_ratio, total_return, max_drawdown

# Prepare data for forecasting (80-20 train-test split)
train_data, test_data, full_series = prepare_data_for_forecasting(df, test_size=0.2)
```

#### `CorrelationAnalyzer` Class
```python
from eda_analysis import CorrelationAnalyzer

# Multi-stock correlation analysis
corr_df = get_multiple_stocks(["AAPL", "MSFT", "GOOGL"], "2y")
analyzer = CorrelationAnalyzer(corr_df)

# Get insights for specific stock
insights = analyzer.get_correlation_insights("AAPL")
# Returns: price_correlations, returns_correlations, 
#          most_correlated, least_correlated, etc.

# Find diversification opportunities
low_corr_pairs = analyzer.find_diversification_pairs(threshold=0.3)
```

### `forecasting.py`

#### `ARIMAForecaster` Class
```python
from forecasting import ARIMAForecaster

# Initialize with automatic order detection
forecaster = ARIMAForecaster(train_data)

# Fit model
info = forecaster.fit()
# Returns: aic, bic, order, summary

# Generate forecast (30 days)
predictions = forecaster.forecast(steps=30)

# Get confidence intervals
conf_int = forecaster.get_confidence_intervals(steps=30, confidence=0.95)
# Returns: forecast, lower_bound, upper_bound

# Evaluate on test data
metrics = forecaster.evaluate(test_data)
# Returns: mse, rmse, mae, mape, r2_score, predictions
```

#### `LSTMForecaster` Class
```python
from forecasting import LSTMForecaster

# Initialize with lookback window
forecaster = LSTMForecaster(train_data, lookback=60)

# Train model
history = forecaster.fit(epochs=50, batch_size=32)
# Returns: loss, val_loss, epochs

# Generate forecast
predictions = forecaster.forecast(steps=30)

# Evaluate on test data
metrics = forecaster.evaluate(test_data)
# Returns: mse, rmse, mae, mape, r2_score, predictions
```

#### `ForecastComparison` Class
```python
from forecasting import ForecastComparison

# Compare all models
comparator = ForecastComparison(train_data, test_data)

# Run individual models
comparator.run_arima()
comparator.run_lstm()
comparator.run_naive_baseline()

# Get comparison summary
comparison_df = comparator.get_comparison_summary()
# Returns: DataFrame with Model, RMSE, MAE, MAPE, R² Score
```

## 📊 Key Metrics Explained

### Volatility
- **Daily Volatility**: Standard deviation of daily returns
- **Rolling Volatility**: 20-day rolling standard deviation
- Higher volatility = Higher risk/reward

### Sharpe Ratio
- Measures risk-adjusted returns
- Formula: (Avg Return - Risk-Free Rate) / Volatility
- Higher ratio = Better risk-adjusted performance
- Assumes 2% annual risk-free rate

### Maximum Drawdown
- Largest peak-to-trough decline during period
- Represents worst-case loss from peak
- Important for risk assessment

### Forecasting Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **RMSE** | √(Σ(actual - pred)²/n) | Lower is better; in original units |
| **MAE** | Σ\|actual - pred\|/n | Mean absolute error; robust to outliers |
| **MAPE** | Σ\|\|actual - pred\|/actual\|/n × 100 | Percentage error; scale-independent |
| **R² Score** | 1 - (SSres/SStot) | Variance explained; 1.0 = perfect fit |

## 📈 Use Cases

### Portfolio Management
- Analyze correlations between holdings
- Identify diversification opportunities
- Monitor sector performance

### Risk Assessment
- Track volatility trends
- Calculate maximum drawdown
- Evaluate risk-adjusted returns (Sharpe ratio)

### Trading Strategy
- Generate price forecasts
- Identify entry/exit signals via RSI
- Support momentum trading strategies

### Investment Research
- Compare multiple stocks simultaneously
- Analyze long-term trends
- Make data-driven decisions

## 💡 Best Practices

### Data Selection
- Use at least 1-2 years of historical data for EDA
- Longer periods (5+ years) for stable trend analysis
- Higher frequency data (daily) for technical indicators

### Model Choice
- **ARIMA**: Use for stationary series or financial returns
- **LSTM**: Use for complex patterns and non-linear relationships
- **Compare**: Always benchmark models before deployment

### Forecasting Advice
- Forecast horizon should be 5-10% of training data length
- Monitor model performance on out-of-sample data
- Update models regularly with new data
- Combine forecasts with fundamental analysis

### JP Morgan Targeting Points
1. **Institutional-Grade Analytics**: Production-ready code with error handling
2. **Multi-Model Approach**: ARIMA/LSTM comparison for robust forecasting
3. **Risk Metrics**: Sharpe ratio, max drawdown, volatility analysis
4. **Correlation Studies**: Portfolio diversification opportunities
5. **Scalable Architecture**: Supports multiple tickers and time periods
6. **Professional Visualizations**: Plotly charts suitable for executive reports

## 🔧 Technical Stack

| Component | Purpose |
|-----------|---------|
| **Streamlit** | Interactive web interface |
| **Plotly** | Advanced visualizations |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computations |
| **yfinance** | Real-time market data |
| **scikit-learn** | ML utilities & metrics |
| **statsmodels** | ARIMA & statistical tests |
| **TensorFlow/Keras** | LSTM neural networks |
| **SciPy** | Scientific computing |

## 🚀 Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **New app** and select your repository
4. Set main file to `forecasting_app.py` or `app.py`
5. Click **Deploy!**

Note: Large ML models (TensorFlow) may require more resources. Consider using a dedicated server for production deployment.

## ⚠️ Disclaimer

This application is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough due diligence and consult with financial advisors before making investment decisions. The forecasts generated should not be considered financial advice.

## 📝 License

Open source - Feel free to modify and distribute.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional forecasting models (Prophet, VAR, GARCH)
- Value at Risk (VaR) calculations
- Options pricing models
- Portfolio optimization algorithms
- Real-time data streaming

---

**Built for JP Morgan & Institutional Finance**
*Professional Stock Market Analytics Platform*
