"""
Python module handling file and data IO.
"""

import pandas as pd
import random

def read_tickers_from_excel(sheet="Stock", exchange="HEL",
        ticker_column="Ticker",
        path="./Yahoo-Ticker-Symbols-September-2017/tickers.xlsx"):
    df = pd.read_excel(path, sheet_name=sheet)
    # Select Exchange == "HEL"
    df = df[df["Exchange"] == exchange]
    df = df[ticker_column]
    # Convert to list
    tickers = df.values.tolist()
    random.shuffle(tickers)
    return tickers

def _accept_ticker(history, ticker):
    """ Criteria, for whether to accept data from a ticker or not.
    """
    if len(history) < 4000:
        print(f"Ticker {ticker} has less than 4000 entries, removing it")
        return False
    # Remove tickers with Stock Splits
    if history["Stock Splits"].sum() > 0:
        print(f"Ticker {ticker} has stock splits, removing it")
        return False
    # Remove tickers with NaN values
    if history.isnull().values.any():
        print(f"Ticker {ticker} has NaN values, removing it")
        return False
    # If the average value of one stock is less than 20 or > 100, remove it
    if history["Close"].mean() < 20 or history["Close"].mean() > 200:
        print(f"""
        Ticker {ticker} has average value {history['Close'].mean()},
        removing it""")
        return False
    return True
  """ Criteria, for whether to accept data from a ticker or not.
  """
  if len(history) < 4000:
    print(f"Ticker {ticker} has less than 4000 entries, removing it")
    return False
  # Remove tickers with Stock Splits
  if history["Stock Splits"].sum() > 0:
    print(f"Ticker {ticker} has stock splits, removing it")
    return False
  # Remove tickers with NaN values at Close column
  if history["Close"].isnull().values.any():
    print(f"Ticker {ticker} has NaN values, removing it")
    return False
  # If the average value of one stock is less than 20 or > 100, remove it
  if history["Close"].mean() < 20 or history["Close"].mean() > 200:
    print(f"""
      Ticker {ticker} has average value {history['Close'].mean()},
      removing it""")
    return False
  return True

def get_histories(tickers, max_tickers=10):
  """
  Returns a dictionary of historical stock data for a given list of tickers.

  Parameters:
  tickers (list): A list of stock tickers to retrieve data for.
  max_tickers (int): The maximum number of tickers to retrieve data for.

  Returns:
  dict: A dictionary of historical stock data for the given tickers.
  """
  histories = {}
  for ticker in tickers.tickers.keys():
    history = tickers.tickers[ticker].history(period="730d", interval="1h")
    #print(history)
    if not _accept_ticker(history, ticker):
      continue
    histories[ticker] = history
    print(f"Added {ticker}")
    if len(histories) >= max_tickers:
      break
  return histories
