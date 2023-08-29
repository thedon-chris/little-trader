import yfinance as yf
import pandas as pd
import numpy as np

def strategy(df, buy_threshold, sell_threshold, ema_span):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    
    # Calculate EMA
    df['EMA'] = df['Close'].ewm(span=ema_span, adjust=False).mean()
    
    # Buy/Sell signals
    df['Signal'] = 0  # Default to no action
    df.loc[(df['RSI'] < buy_threshold) & (df['Close'] > df['EMA']), 'Signal'] = 1  # Buy signal
    df.loc[(df['RSI'] > sell_threshold) & (df['Close'] < df['EMA']), 'Signal'] = -1  # Sell signal
    
    # Calculate returns
    df['Return'] = df['Close'].pct_change() * df['Signal'].shift(1)
    cumulative_return = (df['Return'] + 1).cumprod().iloc[-1] - 1
    
    # Count trades
    trades = df['Signal'].diff().ne(0).sum()

    return cumulative_return, trades

def optimize_parameters(df):
    best_return = float('-inf')
    best_parameters = None
    best_trades = 0

    for buy_threshold in range(5, 100, 5):
        for sell_threshold in range(buy_threshold, 100, 5):  # Ensure sell threshold is always >= buy threshold
            for ema_span in range(5, 50, 5):
                cumulative_return, trades = strategy(df.copy(), buy_threshold, sell_threshold, ema_span)
                if cumulative_return > best_return:
                    best_return = cumulative_return
                    best_parameters = (buy_threshold, sell_threshold, ema_span)
                    best_trades = trades

    return best_parameters, best_return, best_trades

if __name__ == "__main__":
    df = yf.download('NVDA', start="2022-01-01", end="2023-01-01", interval="1h")
    best_parameters, best_return, best_trades = optimize_parameters(df)
    
    print(f"Best Buy Threshold: {best_parameters[0]}, Best Sell Threshold: {best_parameters[1]}, Best EMA Span: {best_parameters[2]}, Best Return: {best_return*100:.2f}%, Total Trades: {best_trades}")

