# 🚀 Quick Start Guide

Get up and running with the Stock Market EDA + Forecasting platform in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- ~500MB free disk space

## Installation (Windows)

### 1. Open PowerShell and Navigate to Project

```powershell
cd "c:\Users\Aayush\OneDrive\Desktop\🚀 Live Stock Market Dashboard\stock-dashboard"
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: First installation may take 5-10 minutes as TensorFlow downloads.

### 4. Run Dashboard

```powershell
streamlit run app.py
```

Or for advanced analytics:

```powershell
streamlit run forecasting_app.py
```

This will open your browser automatically at `http://localhost:8501`

## Quick Demo

### Scenario 1: Basic Dashboard (2 minutes)

1. **Run**: `streamlit run app.py`
2. **Sidebar**: Enter ticker `AAPL`
3. **Select**: Period = "1y", Interval = "1d"
4. **View**: Interactive candlestick chart with moving averages
5. **Compare**: Select MSFT, GOOGL from comparison dropdown

### Scenario 2: EDA Analysis (3 minutes)

1. **Run**: `streamlit run forecasting_app.py`
2. **Tab**: Select "📈 EDA Analysis"
3. **Ticker**: Enter `AAPL`
4. **Period**: Select "2y"
5. **View**: 
   - KPI metrics (price, volatility, sharpe ratio)
   - Technical indicators (MA, Bollinger Bands, RSI)
   - Returns distribution
   - Volatility trends

### Scenario 3: Correlation Analysis (2 minutes)

1. **Tab**: Select "🔗 Correlation Matrix"
2. **Stocks**: Select 5-8 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)
3. **View**:
   - Price correlation heatmap
   - Returns correlation heatmap
   - Diversification opportunities
   - Low-correlation pairs

### Scenario 4: Price Forecasting (5 minutes)

1. **Tab**: Select "🔮 Price Forecasting"
2. **Ticker**: AAPL
3. **Model**: Select "Compare All"
4. **Days**: 30
5. **Click**: "🚀 Run Forecast"
6. **Results**:
   - ARIMA forecast with confidence intervals
   - LSTM forecast
   - Naive baseline comparison
   - Best model recommendation

## Using the Jupyter Notebook

For research and detailed analysis:

```powershell
jupyter notebook Stock_Market_EDA_Forecasting.ipynb
```

**Features**:
- Complete EDA workflow
- Correlation analysis
- ARIMA model training
- LSTM deep learning
- Model comparison
- Publication-ready visualizations

## File Structure Reference

```
stock-dashboard/
├── app.py                              # Main dashboard
├── forecasting_app.py                  # EDA + Forecasting dashboard
├── eda_analysis.py                     # EDA classes (StockEDA, CorrelationAnalyzer)
├── forecasting.py                      # Forecast models (ARIMA, LSTM)
├── utils.py                            # Data fetching utilities
├── Stock_Market_EDA_Forecasting.ipynb  # Interactive notebook
├── requirements.txt                    # Dependencies
├── README.md                           # Full documentation
├── JP_Morgan_Analytics_Guide.md        # Finance reference
└── QUICKSTART.md                       # This file
```

## Common Commands

```powershell
# Activate environment
.\venv\Scripts\Activate

# Install new package
pip install package_name

# Deactivate environment
deactivate

# View installed packages
pip list

# Update all packages
pip install --upgrade pip

# Run dashboard (live update on code changes)
streamlit run app.py --logger.level=debug

# Clear cache
streamlit cache clear
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: 
```powershell
pip install tensorflow -U
```

This creates CPU-optimized LSTM models. For GPU support, install `tensorflow-gpu`.

### Issue: Dashboard opens but shows "No module named 'yfinance'"

**Solution**:
```powershell
pip install yfinance pandas plotly scikit-learn statsmodels
```

### Issue: "Port 8501 already in use"

**Solution**:
```powershell
streamlit run app.py --server.port 8502
```

### Issue: Slow LSTM training on first run

**Solution**: This is normal. TensorFlow compiles on first use. Subsequent runs are faster.

### Issue: "Data not downloading" from yfinance

**Solution**: 
1. Check internet connection
2. Try different ticker (MSFT, GOOGL instead of penny stocks)
3. Use longer history period (avoid weekends/holidays)

## Keyboard Shortcuts (in Streamlit)

| Shortcut | Action |
|----------|--------|
| `r` | Rerun app |
| `c` | Clear cache |
| `s` | Show source code |
| `? ` | Help |

## Tips & Tricks

### Speed Up Dashboard
```python
# Add to app.py top
import streamlit as st
st.set_page_config(layout="wide")  # Wider layout
```

### Export Data to Excel

Modify `forecasting_app.py`:
```python
if st.button("📥 Export Forecast"):
    forecast_df.to_excel("forecast.xlsx")
    st.success("Exported!")
```

### Use Different Data Period

Change in sidebar:
```python
period = st.selectbox("Time Period", 
    options=["1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
```

### Monitor Multiple Tickers

In correlation tab, select 8+ tickers:
```python
tickers = st.multiselect("Select stocks", 
    options=[...], default=["AAPL", "MSFT", ...])
```

## Next Steps

1. **Explore Data**: Run all three apps with different tickers
2. **Read Documentation**: Check [README.md](README.md) and [JP_Morgan_Analytics_Guide.md](JP_Morgan_Analytics_Guide.md)
3. **Run Notebook**: Execute [Stock_Market_EDA_Forecasting.ipynb](Stock_Market_EDA_Forecasting.ipynb)
4. **Customize**: Modify code for your specific needs
5. **Deploy**: See README.md for Streamlit Cloud deployment

## Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **yfinance Docs**: https://github.com/ranaroussi/yfinance
- **ARIMA Tutorial**: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
- **LSTM Guide**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| Load dashboard | 2-3s | First load slower |
| Fetch 1 year data | 1-2s | Depends on internet |
| EDA analysis | 2-5s | Includes all indicators |
| Correlation (8 stocks) | 3-5s | Caching improves repeat |
| ARIMA forecast | 5-10s | Includes model fitting |
| LSTM train (30 epochs) | 30-60s | GPU would be faster |
| Model comparison (all) | 60-90s | Run once, use results |

## Support & Feedback

Having issues? Check these first:
1. Virtual environment activated? (`venv\Scripts\Activate`)
2. All dependencies installed? (`pip list`)
3. Internet connection active? (yfinance needs it)
4. Correct file permissions? (Run as admin if needed)

---

**Ready to get started? Run:**
```powershell
streamlit run forecasting_app.py
```

**Happy analyzing! 📊📈🚀**
