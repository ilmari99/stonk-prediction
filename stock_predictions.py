"""
Main Python script to predict stock evolution based on Yahoo Finance data.
"""

import json
import numpy as np
import os
import locale

# Curve fitting
import tensorflow as tf

# Scaler
from sklearn.preprocessing import MinMaxScaler

# Data Source
import yfinance as yf
import matplotlib.pyplot as plt

from stock_modules.stock_io import (read_tickers_from_excel,
                                    get_histories)
from stock_modules.stock_transform import (histories_to_array,
                                           create_batch_xy)
from stock_modules.stock_ml import create_tf_model

# Figures
from stock_modules.stock_plot import plot_numpy_arr_cols
if not os.path.exists("./figures"):
  os.mkdir("./figures")

ENCODING = locale.getpreferredencoding()
HISTORY_ARRAY_PATH = "./histories_arr.npy"
MODEL_PATH = "./model.h5"
SELECTED_TICKERS_PATH = "./TICKERS_TO_FOLLOW.json"


def read_histories(RENEW_HISTORIES, MAX_TICKERS, SHEET, EXCHANGE, TICKER_COLUMN, custom_tickers = {}):
        """
        Reads stock histories from Yahoo Finance API and returns the selected tickers and their histories as a numpy array.

        Args:
        ENCODING (str): The encoding of the file.
        RENEW_HISTORIES (bool): A boolean value indicating whether to renew the histories or not.
        MAX_TICKERS (int): The maximum number of tickers to follow.
        SHEET (str): The name of the sheet in the excel file.
        EXCHANGE (str): The name of the exchange.
        TICKER_COLUMN (str): The name of the column containing the tickers.

        Returns:
        tuple: A tuple containing the selected tickers and their histories as a numpy array.
        """
        histories_arr = None
        # if either there is no history array or the histories should be renewed
        if not os.path.exists(HISTORY_ARRAY_PATH) or RENEW_HISTORIES:
            # Read tickers from excel
            if not custom_tickers:
                tickers_from_excel = read_tickers_from_excel(
                    sheet=SHEET,
                    exchange=EXCHANGE,
                    ticker_column=TICKER_COLUMN
                )
            # Use custom tickers
            else:
                tickers_from_excel = custom_tickers

            print(f"Fetching data for tickers: {tickers_from_excel}")

            tickers = yf.tickers.Tickers(tickers_from_excel)
            print(f"Tickers found from yfinance: {tickers.tickers.keys()}")
            histories = get_histories(tickers, max_tickers=MAX_TICKERS)
            actually_selected_tickers = list(histories.keys())
            print(f"Selected tickers: {actually_selected_tickers}")
            histories_arr = histories_to_array(histories)
            with open(SELECTED_TICKERS_PATH, "w", encoding=ENCODING) as f:
                json.dump(actually_selected_tickers, f)
            np.save(HISTORY_ARRAY_PATH, histories_arr)
        else:
            actually_selected_tickers = json.load(
                open(SELECTED_TICKERS_PATH, "r", encoding=ENCODING)
            )
            histories_arr = np.load(HISTORY_ARRAY_PATH)
        # Only take max tickers
            histories_arr = histories_arr[:,:min(histories_arr.shape[1],MAX_TICKERS)]
        return actually_selected_tickers,histories_arr



if __name__ == "__main__":
  # Fetch data from Yahoo Finance
  RENEW_HISTORIES = False

  # Train a new model
  RENEW_MODEL = False

  # Show the training data
  SHOW_TRAIN_DS = False

  # Number of lagged hours to use as input
  MHOURS = 24

  # Test set size
  TEST_SIZE = 0.2

  # How many hours to predict into the future based only on past data
  RECURSE_TO = 48

  # Maximum number of tickers to to use
  MAX_TICKERS = 50

  # Training parameters
  EPOCHS = 1000
  BATCH_SIZE = 32
  PATIENCE = 25

  # Which data to use
  SHEET = "Stock"
  EXCHANGE = "HEL"
  TICKER_COLUMN = "Ticker"
  CUSTOM_TICKERS = {}



  actually_read_tickers, histories_arr = read_histories(RENEW_HISTORIES, MAX_TICKERS, SHEET, EXCHANGE, TICKER_COLUMN, CUSTOM_TICKERS)

  IND_CONVERSION = {i:t for i,t in enumerate(actually_read_tickers)}
  print(f"Shape of histories array: {histories_arr.shape}")

  if SHOW_TRAIN_DS:
    ax = plot_numpy_arr_cols(histories_arr[:,:5],
                             ind_conversion=IND_CONVERSION)
    ax.set_title("History of 5 stocks")
    ax.grid()
    plt.savefig("./figures/stock_hist.eps")

  minmax_scaler = MinMaxScaler()
  # Fit the scaler to the first 1-TEST_SIZE of the data
  minmax_scaler = minmax_scaler.fit(
      histories_arr[:-int(histories_arr.shape[0]*TEST_SIZE),:]
  )

  histories_arr = minmax_scaler.transform(histories_arr)

  # Batch X data into sequences of length MHOURS (from T to T+n), and Y data
  # into sequences of length 1 (T+n+1)
  X, Y = create_batch_xy(MHOURS, histories_arr, overlap=True)

  X_og = X.copy()
  Y_og = Y.copy()

  test_sz = int(X.shape[0]*TEST_SIZE)

  # Split the data into train and test sets
  X_train = X[:-test_sz,:,:]
  Y_train = Y[:-test_sz,:]
  X_test = X[-test_sz:,:,:]
  Y_test = Y[-test_sz:,:]

  # Fit model by showing it the data from the last MHOURS hours, and
  # predicting the next hour
  if os.path.exists(MODEL_PATH) and not RENEW_MODEL:
    model = tf.keras.models.load_model(MODEL_PATH)
  else:
    model = create_tf_model(MHOURS, X.shape[2])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
      )
    model.fit(X_train, Y_train,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_test, Y_test), verbose=1,
              callbacks=[early_stop], shuffle=True)
    model.save(MODEL_PATH)
  
  # Predict the next hours for the test data
  Y_pred = model.predict(X_test)

  # Test on the last TEST_SIZE of the data
  test_begin_idx = int(histories_arr.shape[0]*(1-TEST_SIZE))
  X_overlap, Y_overlap = create_batch_xy(
      MHOURS, histories_arr[test_begin_idx-MHOURS:], overlap=True
  )
  # predict every hour
  Y_pred_overlap = model.predict(X_overlap)

  # Invert scaling of predictions
  Y_pred_overlap = minmax_scaler.inverse_transform(Y_pred_overlap)
  Y_overlap = minmax_scaler.inverse_transform(Y_overlap)

  # Plot true values and predicted values for 5 selected stocks
  stock_idxs = np.random.randint(0,X.shape[2],5)
  fig, ax = plt.subplots()
  for stock_idx in stock_idxs:
    ax.plot(Y_overlap[:,stock_idx], label=f"True {IND_CONVERSION[stock_idx]}")
    ax.plot(Y_pred_overlap[:,stock_idx],
            label=f"Predicted {IND_CONVERSION[stock_idx]}")
  ax.legend()
  ax.set_title("True and predicted values for 5 stocks")
  # Save
  plt.savefig("./figures/1h-ahead-predictions.eps")

  # Start from the RECURSE_TO latest hour, predict the next hour, and then use the
  # predicted hour to predict the next hour, and so on
  X_start = X_test[-RECURSE_TO,:,:]
  print(f"X_start shape: {X_start.shape}")
  Y_preds_recurse = X_start[-1,:].reshape(1,X.shape[2])
  Y_true_recurse = Y_test[-RECURSE_TO:,:]
  print(f"Y_preds_recurse shape: {Y_preds_recurse.shape}")
  print(f"Y_true_recurse shape: {Y_true_recurse.shape}")

  for i in range(RECURSE_TO - 1):
    Y_pred = model.predict(X_start.reshape(1,MHOURS,X.shape[2]))
    Y_preds_recurse = np.concatenate([Y_preds_recurse, Y_pred], axis=0)
    # Shift X_start one hour forward
    X_start = np.concatenate([X_start[1:,:], Y_pred], axis=0)
  Y_preds_recurse = np.array(Y_preds_recurse).squeeze()
  print(f"Y_preds_recurse shape: {Y_preds_recurse.shape}")

  # Invert scaling of predictions
  Y_preds_recurse = minmax_scaler.inverse_transform(Y_preds_recurse)
  Y_true_recurse = minmax_scaler.inverse_transform(Y_true_recurse)

  # Select 5 stocks with the smallest mae to plot
  mae_recursive_preds = np.mean(np.abs(Y_preds_recurse - Y_true_recurse),
                                axis=0)
  stock_idxs = np.argsort(mae_recursive_preds)[:10]
  # only take 5 randomly
  stock_idxs = np.random.choice(stock_idxs, 4, replace=False)

  fig, ax = plt.subplots()
  for stock_idx in stock_idxs:
    ax.plot(Y_true_recurse[:,stock_idx],
            label=f"True {IND_CONVERSION[stock_idx]}")
    ax.plot(Y_preds_recurse[:,stock_idx],
            label=f"Predicted {IND_CONVERSION[stock_idx]}")
  ax.legend()
  ax.set_title(f"""
               True and predicted values for 4 stocks up to {RECURSE_TO} hours
               in the future
               """)
  plt.savefig("./figures/recursive_prediction.eps")

  # Calculate the error of the predictions
  mae_hourly_preds = np.mean(np.abs(Y_pred - Y_test), axis=0)
  print(f"Mean absolute error of hourly predictions: {mae_hourly_preds}")
  print(f"Mean absolute error of recursive predictions: {mae_recursive_preds}")

  # As a baseline, calculate how much the price changes per hour on the test
  # data
  average_hourly_change = np.mean(np.abs(Y_overlap[1:] - Y_overlap[:-1]),
                                  axis=0)
  print(f"Average hourly change: {average_hourly_change}")
