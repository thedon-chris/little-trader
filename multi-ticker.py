import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

CASH_INVESTED = 10000  # Example amount

def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
    stock_data['EMA'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    return stock_data

def implement_strategy(data, buy_threshold, sell_threshold):
    data['Position'] = np.where(data['Close'] > data['EMA'], 1, -1)
    data['Strategy_Return'] = data['Close'].pct_change() * data['Position'].shift(1)
    trades = data['Position'].diff().abs().sum() / 2
    return data, trades

def backtest_strategy(ticker, training_start, training_end, test_start, test_end):
    training_data = fetch_data(ticker, training_start, training_end)
    test_data = fetch_data(ticker, test_start, test_end)
    
    max_return = float('-inf')
    best_buy_threshold = None
    best_sell_threshold = None

    for buy_threshold in range(5, 100, 5):
        for sell_threshold in range(5, 100, 5):
            training_data, _ = implement_strategy(training_data.copy(), buy_threshold, sell_threshold)
            strategy_return = (training_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1
            
            if strategy_return > max_return:
                max_return = strategy_return
                best_buy_threshold = buy_threshold
                best_sell_threshold = sell_threshold

    test_data, trades = implement_strategy(test_data.copy(), best_buy_threshold, best_sell_threshold)
    test_return = (test_data['Strategy_Return'] + 1).cumprod().iloc[-1] - 1

    return best_buy_threshold, best_sell_threshold, test_return, trades

def find_similar_stocks(base_ticker, potential_tickers, start_date, end_date):
    base_data = yf.download(base_ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()
    correlated_tickers = {}
    
    for ticker in potential_tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close'].pct_change().dropna()
            correlation = base_data.corr(ticker_data)
            correlated_tickers[ticker] = correlation
        except:
            continue

    # Sort by correlation value
    sorted_tickers = sorted(correlated_tickers.items(), key=lambda x: x[1], reverse=True)
    return [ticker[0] for ticker in sorted_tickers[:3]]  # Return top 3 correlated tickers





if __name__ == "__main__":
    # Define the base ticker and a list of potential tickers
    base_ticker = "NVDA"
    potential_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "AMD", "INTC"]  # Sample tickers, can be expanded

    # Define the time periods
    training_start = "2022-01-01"
    training_end = "2023-01-01"
    test_start = "2023-01-01"
    test_end = "2023-08-01"

    # Find stocks with similar performance to the base ticker
    similar_stocks = find_similar_stocks(base_ticker, potential_tickers, training_start, test_end)

    # Backtest strategy on similar stocks
    for ticker in similar_stocks:
        buy_threshold, sell_threshold, test_return, trades = backtest_strategy(ticker, training_start, training_end, test_start, test_end)
        allocation = CASH_INVESTED * (test_return + 1) / len(similar_stocks)  # Evenly distribute the cash among selected stocks
        print(f"Ticker: {ticker}, Buy Threshold: {buy_threshold}, Sell Threshold: {sell_threshold}, Test Return: {test_return*100:.2f}%, Trades: {trades}, Allocation: ${allocation:.2f}")
