import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
    return stock_data

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

def apply_strategy(data, rsi_buy_threshold, rsi_sell_threshold, ema_span):
    data['EMA'] = data['Close'].ewm(span=ema_span, adjust=False).mean()
    buy_signal = (data['RSI'] < rsi_buy_threshold) & (data['Close'] > data['EMA'])
    sell_signal = (data['RSI'] > rsi_sell_threshold) & (data['Close'] < data['EMA'])
    
    data['Position'] = buy_signal.astype(int) - sell_signal.astype(int)
    data['Strategy_Return'] = data['Close'].pct_change() * data['Position'].shift(1)
    
    trades = data['Position'].diff().abs().sum() // 2
    return data, trades

def simulate_strategy(data, rsi_buy_threshold_range, rsi_sell_threshold_range, ema_span_range):
    max_return = float('-inf')
    best_rsi_buy_threshold = None
    best_rsi_sell_threshold = None
    best_ema_span = None
    best_trades = 0

    for rsi_buy_threshold in rsi_buy_threshold_range:
        for rsi_sell_threshold in rsi_sell_threshold_range:
            for ema_span in ema_span_range:
                strategy_data, trades = apply_strategy(data.copy(), rsi_buy_threshold, rsi_sell_threshold, ema_span)
                strategy_return = (strategy_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
                
                if strategy_return > max_return:
                    max_return = strategy_return
                    best_rsi_buy_threshold = rsi_buy_threshold
                    best_rsi_sell_threshold = rsi_sell_threshold
                    best_ema_span = ema_span
                    best_trades = trades

    return best_rsi_buy_threshold, best_rsi_sell_threshold, best_ema_span, max_return, best_trades

if __name__ == "__main__":
    ticker = "NVDA"
    
    # Define the range for RSI thresholds and EMA span
    rsi_buy_threshold_range = range(5, 100, 5)
    rsi_sell_threshold_range = range(5, 100, 5)
    ema_span_range = range(5, 50, 5)
    
    # Fetch 18 months of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(months=18)).strftime('%Y-%m-%d')
    data = fetch_data(ticker, start_date, end_date)
    data = compute_rsi(data)
    
    # Split data into training and testing sets
    train_data = data.iloc[:int(12/18 * len(data))]
    test_data = data.iloc[int(12/18 * len(data)):]

    # Use training data to find the best parameters
    best_rsi_buy_threshold, best_rsi_sell_threshold, best_ema_span, _, _ = simulate_strategy(
        train_data, rsi_buy_threshold_range, rsi_sell_threshold_range, ema_span_range
    )
    
    # Use the best parameters on the testing data
    test_strategy_data, trades = apply_strategy(test_data.copy(), best_rsi_buy_threshold, best_rsi_sell_threshold, best_ema_span)
    test_return = (test_strategy_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1

    print(f"Best Buy Threshold: {best_rsi_buy_threshold}, Best Sell Threshold: {best_rsi_sell_threshold}, Best EMA Span: {best_ema_span}, Test Return: {test_return*100:.2f}%, Total Trades: {trades}")
