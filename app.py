import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Fetch data
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Calculate RSI
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    
    return data

# Implement strategy
def implement_strategy(data):
    data['Short_SMA'] = data['Close'].rolling(window=50).mean()
    data['Long_SMA'] = data['Close'].rolling(window=200).mean()
    
    # Generate buy/sell signals
    buy_signal = (data['RSI'] < 30) & (data['Short_SMA'] > data['Long_SMA'])
    sell_signal = (data['RSI'] > 70) & (data['Short_SMA'] < data['Long_SMA'])
    
    data['Position'] = np.where(buy_signal, 1, np.where(sell_signal, -1, np.nan))
    data['Position'].fillna(method='ffill', inplace=True)
    
    # Calculate daily returns
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Position'].shift(1)
    
    return data

# Main function
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today's date
    
    data = fetch_data(ticker, start_date, end_date)
    data = compute_rsi(data)
    data = implement_strategy(data)
    
    # Drop NaN values before calculating cumulative returns
    data.dropna(inplace=True)

    # Ensure there's data left after dropping NaNs
    if not data.empty:
        # Calculate cumulative returns
        cumulative_market_returns = (data['Market_Return'] + 1).cumprod() - 1
        cumulative_strategy_returns = (data['Strategy_Return'] + 1).cumprod() - 1

        # Display performance
        print("Cumulative Market Returns as of {}: {:.2f}%".format(end_date, cumulative_market_returns.iloc[-1] * 100))
        print("Cumulative Strategy Returns as of {}: {:.2f}%".format(end_date, cumulative_strategy_returns.iloc[-1] * 100))
    else:
        print("No data left after preprocessing. Check the data fetching and processing steps.")
