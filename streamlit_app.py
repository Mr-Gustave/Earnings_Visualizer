import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe

#python3 -m streamlit run streamlit_app.py

# --- Google Sheets Config ---
@st.cache_data
def load_master_from_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    creds_dict = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)


    client = gspread.authorize(creds)
    sheet = client.open_by_key("19M06LpwDMRQzBgTAewlN8aq300ukLw1xVC-ObX1vcgw").worksheet("Master2025")
    df = get_as_dataframe(sheet, evaluate_formulas=True, header=0)
    df.dropna(how="all", inplace=True)
    df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors='coerce')
    return df

# --- Load Master Table ---
try:
    master_df = load_master_from_google_sheets()
except Exception as e:
    st.error(f"Google Sheets'e erişilemedi: {e}")
    st.stop()

# --- Sidebar: Ticker & Date ---
st.sidebar.title("Earnings Reaction")

tickers = master_df["Ticker"].dropna().unique()
selected_ticker = st.sidebar.selectbox("Select Ticker", sorted(tickers))

dates = master_df[master_df["Ticker"] == selected_ticker]["Report_Date"].dropna().unique()
selected_date = st.sidebar.selectbox("Select Earnings Date", sorted(dates))

# --- EPS & Total Move ---
match = master_df[(master_df["Ticker"] == selected_ticker) & (master_df["Report_Date"] == selected_date)]
if match.empty:
    st.error("Data not found.")
    st.stop()

actual_eps = match["EPS"].values[0] if "EPS" in match.columns else 2.40
estimate_eps = match["Estimate"].values[0] if "Estimate" in match.columns else 2.35
total_move = match["Total Move"].values[0] if "Total Move" in match.columns else None

marker_color = 'green' if actual_eps > estimate_eps else 'red'

# --- YFinance: Data Pull ---
start_date = selected_date - pd.Timedelta(days=60)
end_date = selected_date + pd.Timedelta(days=60)
stock_data = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)

if stock_data.empty:
    st.warning("No stock data.")
    st.stop()

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.droplevel(1)

ohlcv_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()

# --- Find Nearest Trading Date ---
nearest_trading_date = ohlcv_data.index[ohlcv_data.index.get_indexer([selected_date], method='nearest')][0]

# --- Subset 60-day Window ---
subset_data = ohlcv_data.copy()
subset_data['Relative_Days'] = (subset_data.index - nearest_trading_date).days
subset_data = subset_data[(subset_data['Relative_Days'] >= -60) & (subset_data['Relative_Days'] <= 60)]
subset_data['MA7'] = subset_data['Close'].rolling(window=7).mean()

# --- Price Chart ---
fig1, ax1 = plt.subplots(figsize=(12, 5))
for idx, row in subset_data.iterrows():
    color = 'green' if row['Close'] >= row['Open'] else 'red'
    ax1.plot([row['Relative_Days'], row['Relative_Days']], [row['Low'], row['High']], color=color, linewidth=1)
    ax1.plot([row['Relative_Days'], row['Relative_Days']], [row['Open'], row['Close']], color=color, linewidth=5)

# MA7 & marker
ax1.plot(subset_data['Relative_Days'], subset_data['MA7'], color='blue', linewidth=1.5, label='7-Day MA')
# Marker konumu: o günün High değerinin biraz üstü
marker_y = subset_data.loc[nearest_trading_date]['High'] * 1.2
ax1.scatter(0, marker_y, color=marker_color, s=200, marker='o', edgecolor='black', zorder=5)

# Total Move Lines
if total_move:
    earnings_price = subset_data.loc[nearest_trading_date]['Close']
    ax1.axhline(y=earnings_price + total_move, color='purple', linestyle='-', linewidth=1.5, label='+Total Move')
    ax1.axhline(y=earnings_price - total_move, color='purple', linestyle='-', linewidth=1.5, label='-Total Move')
    ax1.axhline(y=earnings_price + total_move / 2, color='gray', linestyle='-', linewidth=1.2, label='+Half Move')
    ax1.axhline(y=earnings_price - total_move / 2, color='gray', linestyle='-', linewidth=1.2, label='-Half Move')

ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax1.set_xticks(np.arange(-60, 61, 10))
ax1.set_xlim(-60, 60)
ax1.set_xlabel('Relative Days')
ax1.set_ylabel('Price ($)')
ax1.set_title(f"{selected_ticker} Price Reaction to Earnings")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()
st.pyplot(fig1)

# --- Volume Chart ---
fig2, ax2 = plt.subplots(figsize=(12, 2.5))
ax2.bar(subset_data['Relative_Days'], subset_data['Volume'] / 1e6, width=1, color='gray', alpha=0.5)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Relative Days')
ax2.set_ylabel('Volume (M)')
ax2.set_title(f"{selected_ticker} Volume Reaction to Earnings")
ax2.set_xticks(np.arange(-60, 61, 10))
ax2.set_xlim(-60, 60)
ax2.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig2)

