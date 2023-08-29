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

# Implement strategy with given RSI thresholds
def implement_strategy(data, rsi_buy_threshold, rsi_sell_threshold):
    data['Short_SMA'] = data['Close'].rolling(window=50).mean()
    data['Long_SMA'] = data['Close'].rolling(window=200).mean()
    
    # Generate buy/sell signals based on RSI thresholds
    buy_signal = (data['RSI'] < rsi_buy_threshold) & (data['Short_SMA'] > data['Long_SMA'])
    sell_signal = (data['RSI'] > rsi_sell_threshold) & (data['Short_SMA'] < data['Long_SMA'])
    
    data['Position'] = np.where(buy_signal, 1, np.where(sell_signal, -1, np.nan))
    data['Position'].fillna(method='ffill', inplace=True)
    
    data['Strategy_Return'] = data['Close'].pct_change() * data['Position'].shift(1)
    
    cumulative_strategy_returns = (data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
    
    return cumulative_strategy_returns

# Main function
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = fetch_data(ticker, start_date, end_date)
    data = compute_rsi(data)
    
    # Test different RSI thresholds
    for rsi_buy_threshold in range(10, 35, 5):  # Testing buy thresholds from 10 to 30 in steps of 5
        for rsi_sell_threshold in range(65, 95, 5):  # Testing sell thresholds from 65 to 90 in steps of 5
            strategy_return = implement_strategy(data.copy(), rsi_buy_threshold, rsi_sell_threshold)
            print(f"RSI Buy Threshold: {rsi_buy_threshold}, RSI Sell Threshold: {rsi_sell_threshold}, Return: {strategy_return*100:.2f}%")
