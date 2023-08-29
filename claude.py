import pandas as pd 
import numpy as np
import datetime
import yfinance as yf

# Parameters
START = "2022-01-01"
END = datetime.datetime.now().strftime("%Y-%m-%d")  

RSI_BUY = range(60, 70)
RSI_SELL = range(75, 85)
SHORT_SMA = [50, 100, 150]  
LONG_SMA = [100, 200, 300]   

LEVERAGE = 2  
STOP_LOSS = -0.05 
TAKE_PROFIT = 0.1

SLIPPAGE = 0.0005 

# Fetch data
def fetch_data(ticker, start, end):
  data = yf.download(ticker, start=start, end=end, interval='1h')
  return data

# Compute RSI
def compute_rsi(data, window=14):
  delta = data['Close'].diff()
  up = delta.clip(lower=0)
  down = -1 * delta.clip(upper=0)
  ema_up = up.ewm(com=window - 1, adjust=False).mean()
  ema_down = down.ewm(com=window - 1, adjust=False).mean()
  rs = ema_up / ema_down
  rsi = 100 - (100 / (1 + rs))
  data['RSI'] = rsi
  return data

# Implement strategy 
def implement_strategy(data, rsi_buy, rsi_sell, short_sma, long_sma):

  data['Short_SMA'] = data['Close'].rolling(window=short_sma).mean()
  data['Long_SMA'] = data['Close'].rolling(window=long_sma).mean() 

  buy_signal = (data['RSI'] < rsi_buy) & (data['Short_SMA'] > data['Long_SMA'])
  sell_signal = (data['RSI'] > rsi_sell) & (data['Short_SMA'] < data['Long_SMA'])

  data['Position'] = np.where(buy_signal, LEVERAGE, np.where(sell_signal, -LEVERAGE, 0))
  
  data['Strategy'] = data['Position'].shift(1) * data['Close']
  data.dropna(inplace=True)

  trades = len(data['Position'].dropna())
  
  return data, trades
  
# Calculate returns  
def calculate_return(data, capital, slippage):

  data['Returns'] = data['Strategy'].pct_change()  
  
  # Fill NA values
  data['Returns'] = data['Returns'].clip(-0.5, 0.5)

  data['Cash'] = capital * (1 + data['Returns']).cumprod()

  data['Cash'] = data['Cash'] - data['Position'].shift(1) * slippage

  return data['Cash'].iloc[-1]

# Optimization
def optimize_strategy(ticker, start, end):

  best_ret = float("-inf")
  best_params = None

  for rsi_b in RSI_BUY:
    for rsi_s in RSI_SELL:
      for sma_s in SHORT_SMA:
        for sma_l in LONG_SMA:
            
          data = fetch_data(ticker, start, end)
          data = compute_rsi(data)
          strategy, trades = implement_strategy(data, rsi_b, rsi_s, sma_s, sma_l)
          ret = calculate_return(strategy, capital=100000, slippage=SLIPPAGE)
            
          if ret > best_ret:
            best_ret = ret
            best_params = (rsi_b, rsi_s, sma_s, sma_l)

  print("Best Parameters:", best_params)
  print("Best Return:", best_ret)

  return best_params


if __name__ == "__main__":
    
  best_params = optimize_strategy("AAPL", START, END)

  data = fetch_data("AAPL", START, END)
  data = compute_rsi(data)
  strategy, trades = implement_strategy(data, *best_params)

  print(calculate_return(strategy, capital=100000, slippage=SLIPPAGE))