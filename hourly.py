import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import multiprocessing


# Fetch hourly data
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
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

# Define a function that will perform the search over a subset of RSI thresholds
def search_best_rsi(rsi_range, data):
    max_return = float('-inf')
    best_rsi_combination = (None, None)
    best_trade_count = 0

    for rsi_buy_threshold in rsi_range:
        for rsi_sell_threshold in rsi_range:
            strategy_data, trade_count = implement_strategy(data.copy(), rsi_buy_threshold, rsi_sell_threshold)
            strategy_return = (strategy_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1

            if strategy_return > max_return:
                max_return = strategy_return
                best_rsi_combination = (rsi_buy_threshold, rsi_sell_threshold)
                best_trade_count = trade_count

    return best_rsi_combination, max_return, best_trade_count


def implement_strategy(data, rsi_buy_threshold, rsi_sell_threshold, short_sma_period=20, long_sma_period=100):
    data['Short_SMA'] = data['Close'].rolling(window=short_sma_period).mean()
    data['Long_SMA'] = data['Close'].rolling(window=long_sma_period).mean()
    
    # Generate buy/sell signals based on RSI thresholds
    buy_signal = (data['RSI'] < rsi_buy_threshold) & (data['Short_SMA'] > data['Long_SMA'])
    sell_signal = (data['RSI'] > rsi_sell_threshold) & (data['Short_SMA'] < data['Long_SMA'])
    
    data['Position'] = np.where(buy_signal, 1, np.where(sell_signal, -1, np.nan))
    data['Position'].fillna(method='ffill', inplace=True)
    
    data['Strategy_Return'] = data['Close'].pct_change() * data['Position'].shift(1)
    
    # Count the number of trades
    trades = data['Position'].diff().abs().sum() // 2
    return data, trades

# Calculate cumulative return at n months from start
def calculate_return_at_month(data, n):
    end_date = data.index[0] + timedelta(days=n*30)  # Assuming a month has 30 days
    if end_date > data.index[-1]:  # If n months exceeds data range
        end_date = data.index[-1]
    monthly_data = data.loc[:end_date]
    return (monthly_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1


def test_stop_loss_range(data, rsi_buy_threshold, rsi_sell_threshold, stop_loss_range):
    results = {}
    
    for stop_loss_pct in stop_loss_range:
        strategy_data = implement_strategy(data.copy(), rsi_buy_threshold, rsi_sell_threshold, stop_loss_pct)
        strategy_return = (strategy_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
        results[stop_loss_pct] = strategy_return
        
    return results

def search_best_rsi_wrapper(args):
    r, data = args
    return search_best_rsi(r, data)

# Main function
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Add this line

    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # For NVDA
    ticker = "NVDA"
    data = fetch_data(ticker, start_date, end_date)
    data = compute_rsi(data)

    max_return = float('-inf')
    best_rsi_combination = (None, None)
    best_trade_count = 0

    # Split the RSI range into chunks
    cpu_count = multiprocessing.cpu_count()
    chunk_size = (95 // cpu_count) + 1
    rsi_ranges = [range(i, i + chunk_size) for i in range(5, 100, chunk_size)]


    # Create argument tuples
    args = [(r, data) for r in rsi_ranges]

    # Initialize a Pool with cpu_count processes
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Use the Pool's map method with the named function
        results = pool.map(search_best_rsi_wrapper, args)

    # Extract the best result from the results
    best_result = max(results, key=lambda x: x[1])

    print(f"For {ticker}, best RSI Buy: {best_result[0][0]}, RSI Sell: {best_result[0][1]}, Return: {best_result[1]*100:.2f}%, Trades: {best_result[2]}")
