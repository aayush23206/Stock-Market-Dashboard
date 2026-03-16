"""
Streamlit app for stock market EDA and forecasting.
Features: Correlation analysis, ARIMA/LSTM forecasting, and performance metrics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils import get_stock_data, get_multiple_stocks
from eda_analysis import StockEDA, CorrelationAnalyzer, prepare_data_for_forecasting
from forecasting import ARIMAForecaster, ForecastComparison, TENSORFLOW_AVAILABLE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock EDA & Forecasting", layout="wide", page_icon="📊")
st.title("📊 Stock Market EDA & Forecasting")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
tab_selection = st.sidebar.radio(
    "Select Analysis",
    options=["📈 EDA Analysis", "🔗 Correlation Matrix", "🔮 Price Forecasting"]
)

ticker_input = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Time Period", options=["1y", "2y", "5y", "max"], index=0)

# ==================== EDA ANALYSIS TAB ====================
if tab_selection == "📈 EDA Analysis":
    st.header(f"Exploratory Data Analysis: {ticker_input}")
    
    with st.spinner(f"Fetching data for {ticker_input}..."):
        df = get_stock_data(ticker_input, period, "1d")
    
    if df.empty:
        st.error(f"Failed to fetch data for {ticker_input}. Please check the ticker symbol.")
    else:
        # Initialize EDA analyzer
        eda = StockEDA(df, ticker_input)
        stats = eda.get_summary_stats()
        
        # Display KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${stats['current_price']:.2f}")
        with col2:
            st.metric("Avg Price", f"${stats['avg_price']:.2f}")
        with col3:
            st.metric("Volatility", f"{stats['daily_volatility']:.2%}")
        with col4:
            st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
        with col5:
            st.metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
        
        st.divider()
        
        # Price chart with technical indicators
        st.subheader("📈 Price Chart with Technical Indicators")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            visible=True
        ))
        
        # Moving averages
        if 'MA_20' in eda.df.columns:
            fig.add_trace(go.Scatter(
                x=eda.df.index,
                y=eda.df['MA_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'MA_50' in eda.df.columns:
            fig.add_trace(go.Scatter(
                x=eda.df.index,
                y=eda.df['MA_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='red', width=1)
            ))
        
        # Bollinger Bands
        if 'BB_Upper' in eda.df.columns:
            fig.add_trace(go.Scatter(
                x=eda.df.index,
                y=eda.df['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(0,100,200,0)'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=eda.df.index,
                y=eda.df['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(0,100,200,0)'),
                fill='tonexty',
                fillcolor='rgba(0,100,200,0.1)',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'{ticker_input} Price with Technical Indicators',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_returns = px.histogram(
                eda.df['Returns'].dropna(),
                nbins=50,
                title='Daily Returns Distribution',
                labels={'value': 'Returns', 'count': 'Frequency'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            fig_volatility = go.Figure()
            fig_volatility.add_trace(go.Scatter(
                x=eda.df.index,
                y=eda.df['Volatility'] * 100,
                mode='lines',
                name='Volatility',
                line=dict(color='#EF553B')
            ))
            fig_volatility.update_layout(
                title='Rolling 20-Day Volatility',
                yaxis_title='Volatility (%)',
                xaxis_title='Date',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_volatility, use_container_width=True)
        
        # RSI indicator
        st.subheader("RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=eda.df.index,
            y=eda.df['RSI'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='purple')
        ))
        
        # Add overbought/oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        fig_rsi.update_layout(
            title=f'{ticker_input} RSI Indicator',
            yaxis_title='RSI',
            xaxis_title='Date',
            template='plotly_white',
            yaxis=dict(range=[0, 100]),
            height=400
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        summary_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Avg Daily Return',
                'Daily Volatility',
                'Min Price',
                'Max Price',
                'Std Dev',
                'Sharpe Ratio',
                'Max Drawdown'
            ],
            'Value': [
                f"{stats['total_return']:.2%}",
                f"{stats['avg_daily_return']:.4%}",
                f"{stats['daily_volatility']:.2%}",
                f"${stats['min_price']:.2f}",
                f"${stats['max_price']:.2f}",
                f"${stats['std_dev']:.2f}",
                f"{stats['sharpe_ratio']:.2f}",
                f"{stats['max_drawdown']:.2%}"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==================== CORRELATION ANALYSIS TAB ====================
elif tab_selection == "🔗 Correlation Matrix":
    st.header("Correlation Analysis")
    
    # Select comparison stocks
    default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    tickers_to_compare = st.multiselect(
        "Select stocks to compare (max 8)",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "WFC"],
        default=default_stocks[:5]
    )[:8]
    
    if tickers_to_compare:
        with st.spinner("Fetching correlation data..."):
            multi_df = get_multiple_stocks(tickers_to_compare, period)
        
        if not multi_df.empty:
            # Initialize correlation analyzer
            corr_analyzer = CorrelationAnalyzer(multi_df)
            
            # Display correlation heatmap
            st.subheader("📊 Price Correlation Heatmap")
            fig_corr = px.imshow(
                corr_analyzer.get_correlation_matrix(),
                text_auto='.2f',
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Stock Price Correlation Matrix'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Returns correlation
            st.subheader("📊 Returns Correlation Heatmap")
            fig_returns_corr = px.imshow(
                corr_analyzer.get_returns_correlation_matrix(),
                text_auto='.2f',
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Daily Returns Correlation Matrix'
            )
            st.plotly_chart(fig_returns_corr, use_container_width=True)
            
            # Main stock correlation insights
            st.subheader(f"Correlation Analysis: {ticker_input}")
            if ticker_input in tickers_to_compare:
                insights = corr_analyzer.get_correlation_insights(ticker_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Most Correlated Stock**")
                    if insights['most_correlated']:
                        st.metric(
                            insights['most_correlated'],
                            f"{insights['most_correlated_value']:.3f}"
                        )
                
                with col2:
                    st.write("**Least Correlated Stock**")
                    st.metric(
                        insights['least_correlated'],
                        f"{insights['least_correlated_value']:.3f}"
                    )
                
                # Diversification opportunities
                st.subheader("🎯 Diversification Opportunities (Low Correlation Pairs)")
                diversific_pairs = corr_analyzer.find_diversification_pairs(threshold=0.4)
                
                if diversific_pairs:
                    pairs_df = pd.DataFrame(
                        diversific_pairs,
                        columns=['Stock 1', 'Stock 2', 'Correlation']
                    )
                    st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No low-correlation pairs found.")
        else:
            st.error("Failed to fetch data for selected tickers.")

# ==================== FORECASTING TAB ====================
elif tab_selection == "🔮 Price Forecasting":
    st.header(f"Price Forecasting: {ticker_input}")
    
    with st.spinner(f"Fetching historical data..."):
        df = get_stock_data(ticker_input, period, "1d")
    
    if df.empty:
        st.error(f"Failed to fetch data for {ticker_input}.")
    else:
        # Prepare data
        train_data, test_data, full_series = prepare_data_for_forecasting(df, test_size=0.2)
        
        # Model selection and parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_options = ["ARIMA"]
            if TENSORFLOW_AVAILABLE:
                model_options.extend(["LSTM", "Compare All"])
            else:
                st.info("💡 TensorFlow not available - LSTM disabled. Using ARIMA only.")
            
            forecast_model = st.selectbox("Forecasting Model", model_options)
        
        with col2:
            forecast_days = st.number_input("Forecast Days", min_value=5, max_value=90, value=30)
        
        with col3:
            if forecast_model == "LSTM":
                lookback = st.number_input("Lookback Window", min_value=10, max_value=180, value=60)
        
        st.divider()
        
        # Run forecasting
        if st.button("🚀 Run Forecast", use_container_width=True):
            with st.spinner("Training model and generating forecast..."):
                
                if forecast_model == "ARIMA":
                    try:
                        forecaster = ARIMAForecaster(train_data)
                        forecaster.fit()
                        forecast_values = forecaster.forecast(forecast_days)
                        conf_intervals = forecaster.get_confidence_intervals(forecast_days, confidence=0.95)
                        
                        # Display results
                        st.success("✅ ARIMA Forecast Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Model", "ARIMA")
                        with col2:
                            st.metric("Next Price (Avg)", f"${forecast_values[-1]:.2f}")
                        with col3:
                            st.metric("Forecast Days", forecast_days)
                        with col4:
                            pct_change = ((forecast_values[-1] - full_series.iloc[-1]) / full_series.iloc[-1]) * 100
                            st.metric("Expected Change", f"{pct_change:+.2f}%", delta=pct_change)
                        
                        # Visualization
                        future_dates = pd.date_range(
                            start=full_series.index[-1],
                            periods=forecast_days + 1,
                            freq='D'
                        )[1:]
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=full_series.index,
                            y=full_series.values,
                            mode='lines',
                            name='Historical Price',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast_values,
                            mode='lines',
                            name='ARIMA Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=conf_intervals['upper_bound'],
                            mode='lines',
                            name='95% CI Upper',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=conf_intervals['lower_bound'],
                            mode='lines',
                            name='95% Confidence Interval',
                            line=dict(color='rgba(0,0,0,0)'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)'
                        ))
                        
                        fig.update_layout(
                            title=f'{ticker_input} ARIMA Forecast ({forecast_days} days)',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            template='plotly_white',
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model evaluation
                        st.subheader("Model Evaluation")
                        eval_metrics = forecaster.evaluate(test_data)
                        
                        eval_df = pd.DataFrame({
                            'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R² Score'],
                            'Value': [
                                f"${eval_metrics['rmse']:.2f}",
                                f"${eval_metrics['mae']:.2f}",
                                f"{eval_metrics['mape']:.2f}%",
                                f"{eval_metrics['r2_score']:.4f}"
                            ]
                        })
                        st.dataframe(eval_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Error in ARIMA forecasting: {str(e)}")
                
                elif forecast_model == "LSTM":
                    if not TENSORFLOW_AVAILABLE:
                        st.error("⚠️ TensorFlow is not available. Please use ARIMA or install TensorFlow locally.")
                    else:
                        try:
                            from forecasting import LSTMForecaster
                            forecaster = LSTMForecaster(train_data, lookback=lookback)
                            fit_info = forecaster.fit(epochs=30)
                            forecast_values = forecaster.forecast(forecast_days)
                            
                            st.success("✅ LSTM Forecast Complete!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Model", "LSTM")
                            with col2:
                                st.metric("Next Price (Avg)", f"${forecast_values[-1]:.2f}")
                            with col3:
                                st.metric("Forecast Days", forecast_days)
                            with col4:
                                pct_change = ((forecast_values[-1] - full_series.iloc[-1]) / full_series.iloc[-1]) * 100
                                st.metric("Expected Change", f"{pct_change:+.2f}%", delta=pct_change)
                            
                            # Visualization
                            future_dates = pd.date_range(
                                start=full_series.index[-1],
                                periods=forecast_days + 1,
                                freq='D'
                            )[1:]
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=full_series.index,
                                y=full_series.values,
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='green')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast_values,
                                mode='lines',
                                name='LSTM Forecast',
                                line=dict(color='orange', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'{ticker_input} LSTM Forecast ({forecast_days} days)',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                template='plotly_white',
                                hovermode='x unified',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Model evaluation
                            st.subheader("Model Evaluation")
                            eval_metrics = forecaster.evaluate(test_data)
                            
                            eval_df = pd.DataFrame({
                                'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R² Score'],
                                'Value': [
                                    f"${eval_metrics['rmse']:.2f}",
                                    f"${eval_metrics['mae']:.2f}",
                                    f"{eval_metrics['mape']:.2f}%",
                                    f"{eval_metrics['r2_score']:.4f}"
                                ]
                            })
                            st.dataframe(eval_df, use_container_width=True, hide_index=True)
                            
                        except Exception as e:
                            st.error(f"Error in LSTM forecasting: {str(e)}")

                
                elif forecast_model == "Compare All":
                    if not TENSORFLOW_AVAILABLE:
                        st.error("⚠️ TensorFlow is not available. 'Compare All' requires LSTM. Using ARIMA only.")
                        # Fallback to ARIMA comparison
                        try:
                            forecaster = ARIMAForecaster(train_data)
                            forecaster.fit()
                            forecast_values = forecaster.forecast(forecast_days)
                            
                            st.success("✅ ARIMA Forecast Complete!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Model", "ARIMA")
                            with col2:
                                st.metric("Next Price (Avg)", f"${forecast_values[-1]:.2f}")
                            with col3:
                                st.metric("Forecast Days", forecast_days)
                            with col4:
                                pct_change = ((forecast_values[-1] - full_series.iloc[-1]) / full_series.iloc[-1]) * 100
                                st.metric("Expected Change", f"{pct_change:+.2f}%", delta=pct_change)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        try:
                            comparator = ForecastComparison(train_data, test_data)
                            
                            st.info("Running all forecasting models...")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                with st.spinner("ARIMA..."):
                                    comparator.run_arima()
                                    st.success("ARIMA ✓")
                            
                            with col2:
                                with st.spinner("LSTM..."):
                                    comparator.run_lstm()
                                    st.success("LSTM ✓")
                            
                            with col3:
                                with st.spinner("Naive Baseline..."):
                                    comparator.run_naive_baseline()
                                    st.success("Baseline ✓")
                            
                            st.success("✅ All Forecasts Complete!")
                            
                            # Comparison summary
                            st.subheader("Model Comparison Summary")
                            comparison_df = comparator.get_comparison_summary()
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                            
                            # Best model recommendation
                            best_model = comparison_df.iloc[0]
                            st.success(
                                f"### 🏆 Best Model: {best_model['Model']} "
                                f"(RMSE: ${best_model['RMSE']:.2f})"
                            )
                            
                        except Exception as e:
                            st.error(f"Error in model comparison: {str(e)}")

st.sidebar.divider()
st.sidebar.info(
    "💡 **Tips:**\n"
    "- Use 2-5 years of data for best results\n"
    "- ARIMA works well for stationary data\n"
    "- LSTM captures complex patterns\n"
    "- Compare models to find best fit"
)
