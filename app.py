import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import get_stock_data, get_stock_info, get_multiple_stocks

st.set_page_config(page_title="Live Stock Market Dashboard", layout="wide", page_icon="📈")

st.title("📈 Live Stock Market Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Controls")
ticker_input = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Time Period", options=["1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=4)
interval = st.sidebar.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
chart_type = st.sidebar.radio("Main Chart Type", options=["Candlestick", "Line"])

compare_tickers = st.sidebar.multiselect(
    "Compare with (select up to 5)",
    options=["MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "RELIANCE.NS", "TCS.NS"],
    default=[]
)

# Render main content
if ticker_input:
    with st.spinner("Fetching data..."):
        df = get_stock_data(ticker_input, period, interval)
        info = get_stock_info(ticker_input)

    if df.empty:
        st.error(f"Failed to fetch data for {ticker_input}. Please check the ticker symbol.")
    else:
        # --- KPI CARDS ---
        col1, col2, col3, col4, col5 = st.columns(5)
        
        def format_currency(val):
            if isinstance(val, (int, float)):
                return f"${val:,.2f}" if "NS" not in ticker_input else f"₹{val:,.2f}"
            return val
            
        def format_number(val):
            if isinstance(val, (int, float)):
                if val >= 1e12:
                    return f"{val/1e12:.2f}T"
                elif val >= 1e9:
                    return f"{val/1e9:.2f}B"
                elif val >= 1e6:
                    return f"{val/1e6:.2f}M"
                return f"{val:,.2f}"
            return val

        current_price = format_currency(info.get("current_price", "N/A"))
        high_52 = format_currency(info.get("52w_high", "N/A"))
        low_52 = format_currency(info.get("52w_low", "N/A"))
        pe_ratio = format_number(info.get("pe_ratio", "N/A"))
        sector = info.get("sector", "N/A")
        name = info.get("name", ticker_input)

        col1.metric("Current Price", current_price)
        col2.metric("52W High", high_52)
        col3.metric("52W Low", low_52)
        col4.metric("P/E Ratio", pe_ratio)
        col5.metric("Sector", sector)

        st.subheader(f"{name} ({ticker_input})")

        # --- MAIN CHART ---
        fig_main = go.Figure()
        
        if chart_type == "Candlestick":
            fig_main.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"))
        else:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close Price"))
        
        fig_main.update_layout(
            template="plotly_dark", 
            title=f"{ticker_input} {chart_type} Chart", 
            xaxis_title="Date", 
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # Layout for Volume and Moving Averages
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            # --- VOLUME CHART ---
            st.subheader("Trading Volume")
            fig_vol = px.bar(df, x=df.index, y='Volume', title="")
            fig_vol.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Volume", height=400)
            st.plotly_chart(fig_vol, use_container_width=True)

        with row2_col2:
            # --- MOVING AVERAGES ---
            st.subheader("Moving Averages (20 & 50 periods)")
            df_ma = df.copy()
            df_ma['MA20'] = df_ma['Close'].rolling(window=20).mean()
            df_ma['MA50'] = df_ma['Close'].rolling(window=50).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df_ma.index, y=df_ma['Close'], mode='lines', name='Close Price', line=dict(color='white', width=1)))
            fig_ma.add_trace(go.Scatter(x=df_ma.index, y=df_ma['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=2)))
            fig_ma.add_trace(go.Scatter(x=df_ma.index, y=df_ma['MA50'], mode='lines', name='MA50', line=dict(color='blue', width=2)))
            fig_ma.update_layout(template="plotly_dark", title="", xaxis_title="Date", yaxis_title="Price", height=400)
            st.plotly_chart(fig_ma, use_container_width=True)

        # --- COMPARISON CHART ---
        if compare_tickers:
            st.subheader("Stock Comparison (Normalized to Base 100)")
            all_tickers = [ticker_input] + compare_tickers
            with st.spinner("Fetching comparison data..."):
                comp_df = get_multiple_stocks(all_tickers, period)
            
            if not comp_df.empty:
                # Normalize to base 100
                normalized_df = (comp_df / comp_df.iloc[0]) * 100
                
                # Reshape for Plotly Express
                # Need to drop any missing columns
                normalized_df = normalized_df.dropna(axis=1, how='all')
                normalized_df = normalized_df.reset_index()
                # Determine the appropriate date column name created by reset_index
                date_col = normalized_df.columns[0]
                normalized_df = normalized_df.melt(id_vars=date_col, var_name='Ticker', value_name='Normalized Price')
                
                fig_comp = px.line(normalized_df, x=date_col, y='Normalized Price', color='Ticker', title="")
                fig_comp.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Normalized Price (Base 100)", height=500)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Could not fetch comparison data. Please check the tickers.")

        # --- RAW DATA TABLE ---
        with st.expander("View Raw Data (Last 30 periods)"):
            st.dataframe(df.tail(30).sort_index(ascending=False), use_container_width=True)
else:
    st.info("Please enter a valid stock ticker in the sidebar to begin.")
