""" This file contains functions for creating a hypothetical trading strategy
based on a model that predicts the price of stocks one hour ahead.
Also possible to create a hypothetically optimal trading strategy, that assumes
price is known one hour ahead.
"""
import numpy as np
from stock_modules.stock_transform import create_batch_xy



def calculate_optimal_invest_strategy(data : np.ndarray) -> np.ndarray:
  """ Calculates the optimaltrading mask for the data. This assumes that we know 1h in to the future.
  The strategy:
  If we are not holding, and price at T+1 is higher than price at T, we buy 1 (marked as -1)
  If we are holding, and price at T+1 is lower than price at T, we sell 1 (marked as 1)
  """
  mask = np.zeros(data.shape)
  for stock_idx in range(data.shape[1]):
     stock_data = data[:,stock_idx]
     holding_stock = False
     for i in range(stock_data.shape[0]):
        # If we are not holding the stock, buy if the price is the lowest so far
        if not holding_stock and stock_data[i] == np.min(stock_data[i:]):
          mask[i,stock_idx] = -1
          holding_stock = True
        # If we are holding the stock, sell if the price is the highest so far
        elif holding_stock and stock_data[i] == np.max(stock_data[i:]):
          mask[i,stock_idx] = 1
          holding_stock = False
        if holding_stock:
          mask[-1,stock_idx] = 1
  return mask

def calculate_profit_on_invest_strategy(data : np.ndarray, mask : np.ndarray) -> float:
  """ Calculate how much profit would be made by using the mask to buy and sell stocks.
  For an individual stock, the profit is the dot product of the mask and the stock price (assuming mask is correct).
  """
  profit = 0
  for stock_idx in range(data.shape[1]):
      profit += np.dot(mask[:,stock_idx], data[:,stock_idx])
  return profit

def test_invest_strategy(data : np.ndarray, window_hours : int, model) -> np.ndarray:
  """ Calculates how much profit would be made by using the model to predict the price of stocks one hour ahead.
  At the beginning we are not holding.
  If we are not holding, and price at T+1 is higher than price at T, we buy 1 (marked as -1)
  If we are holding, and price at T+1 is lower than price at T, we sell 1 (marked as 1)

  At each point the price at T+1 is predicted using the last window_hours hours of data.

  We return a mask (1 for buy, -1 for sell, 0 for hold) with size (data.shape[0],data.shape[1]) telling when to buy and sell.
  """
  mask = np.zeros(data.shape)
  X, Y = create_batch_xy(window_hours, data, overlap=True)
  # Predict the next hours for the data
  Y_pred = model.predict(X)
  # We are not holding any stocks at the beginning
  holding_bool = np.zeros(data.shape[1])
  # Loop through the last hour of X and the model predictions
  for i in range(X.shape[0]):
     current_prices = X[i,-1,:]
     predicted_prices = Y_pred[i,:]
     # Check all stocks
     for stock_idx in range(data.shape[1]):
        # if we are not holding a stock and price at T+1 is higher than price at T, buy
        if holding_bool[stock_idx] == 0 and predicted_prices[stock_idx] > current_prices[stock_idx]:
          mask[i,stock_idx] = -1
          holding_bool[stock_idx] = 1
        # if we are holding a stock and price at T+1 is lower than price at T, sell
        elif holding_bool[stock_idx] == 1 and predicted_prices[stock_idx] < current_prices[stock_idx]:
          mask[i,stock_idx] = 1
          holding_bool[stock_idx] = 0
  # Sell all remaining stocks at the end if we have any
  for stock_idx in range(data.shape[1]):
    if holding_bool[stock_idx] == 1:
      mask[-1,stock_idx] = 1
  return mask