import json
import numpy as np
import pandas as pd
import random
import os
from TimeSeriesDataset import TimeSeriesDataset
# curve fitting
import tensorflow as tf
# random forest
from sklearn.ensemble import RandomForestRegressor
# Scaler
from sklearn.preprocessing import MinMaxScaler
#Data Source
import yfinance as yf
import matplotlib.pyplot as plt

def read_tickers_from_excel(sheet="Stock", exchange="HEL", ticker_column="Ticker", path="./Yahoo-Ticker-Symbols-September-2017/tickers.xlsx"):
    df = pd.read_excel(path, sheet_name=sheet)
    # Select Exchange == "HEL"
    df = df[df["Exchange"] == exchange]
    df = df[ticker_column]
    # Convert to list
    tickers = df.values.tolist()
    random.shuffle(tickers)
    return tickers

def get_histories(tickers,max_tickers=10):
    # Get the historical data for each stock
    histories = {}
    for ticker in tickers.tickers.keys():
        history = tickers.tickers[ticker].history(period="730d", interval="1h")
        # Remove tickers with less than 4000 entries
        if len(history) < 4000:
            print(f"Ticker {ticker} has less than 2000 entries, removing it")
            continue
        # Remove tickers with Stock Splits
        if history["Stock Splits"].sum() > 0:
            print(f"Ticker {ticker} has stock splits, removing it")
            continue
        # Remove tickers with NaN values
        if history.isnull().values.any():
            print(f"Ticker {ticker} has NaN values, removing it")
            continue
        # If the average value of one stock is less than 1 or > 1000, remove it
        if history["Close"].mean() < 20 or history["Close"].mean() > 100:
            print(f"Ticker {ticker} has average value {history['Close'].mean()}, removing it")
            continue
        histories[ticker] = history
        print(f"Added {ticker}")
        if len(histories) >= max_tickers:
            break
    return histories

def histories_to_array(histories):
    """Convert a dictionary of histories to a 2D array
    """
    values = []
    # Cutoff, so all histories have the same length
    cutoff = min([len(histories[ticker]) for ticker in histories])
    for ticker in histories:
        values.append(histories[ticker]["Close"].values[:cutoff])
    values = np.array(values).T
    return values

def plot_numpy_arr_cols(arr, ax=None, ind_conversion={}):
    """ Plot the columns of a 2d numpy array.
    If ax is None, create a new figure.
    ind_conversion is a dictionary mapping column indices to names, which will be used as labels.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for col_id in range(arr.shape[1]):
        if col_id in ind_conversion:
            label = ind_conversion[col_id]
        else:
            label = ""
        ax.plot(arr[:,col_id], label=label)
    return ax

def create_stateful_tf_model(input_shape, batch_size = 1):
    """ Create a LSTM model which takes in the values of multiple stocks in the last hour, and
    outputs the predicted (residual) value for each stock in the next time step.
    Input shape is (N,1) and output shape is (N,1).
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, stateful=True, input_shape=input_shape,batch_size = batch_size),
        tf.keras.layers.LSTM(16, return_sequences=True, stateful=True),
        tf.keras.layers.LSTM(8, return_sequences=True, stateful=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        # Output (N,1)
        tf.keras.layers.Dense(input_shape[0], activation='linear'),
        tf.keras.layers.Reshape((input_shape[0],))

    ])
    print(f"Model summary: {model.summary()}")
    # Compile the model
    model.compile(optimizer='adam', loss='mae')
    return model

def create_tf_model(M, N):
    """ Create a LSTM model which takes in M values of N stocks (MxN), and
    outputs the predicted value for each stock in the next time step (1 x N).
    M is the batch size, and N is the number of stocks.
    """
    inputs = tf.keras.layers.Input(shape=(M,N))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(16, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(8, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    # Output (1,N)
    outputs = tf.keras.layers.Dense(N, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model


def plot_dataset(ds : TimeSeriesDataset, index_conversion):
    """ Plot the dataset ds, with index conversion ind_conversion
    """
    axes = []
    fig,ax = plt.subplots()
    plot_numpy_arr_cols(ds.residual_array, ax=ax, ind_conversion=index_conversion)
    ax.set_title("Residual data")
    axes.append((fig,ax))
    fig,ax = plt.subplots()
    plot_numpy_arr_cols(ds.trend_array, ax=ax, ind_conversion=index_conversion)
    ax.set_title("Trend data")
    axes.append((fig,ax))
    fig,ax = plt.subplots()
    plot_numpy_arr_cols(ds.seasonality_array, ax=ax, ind_conversion=index_conversion)
    ax.set_title("Seasonal data")
    axes.append((fig,ax))
    return axes

TICKERS_TO_FOLLOW = []
def create_batch_XY(MHOURS, histories_arr, overlap= False):
    """
    Batch X data into sequences of length MHOURS (from T to T+n), and Y data into sequences of length 1 (T+n+1)

    Args:
        MHOURS (int): The number of hours to include in each input sequence.
        histories_arr (numpy.ndarray): A 2D array of historical stock prices, where each row represents a time step and each column represents a different stock.

    """
    X = []
    Y = []
    for i in range(0,histories_arr.shape[0]-MHOURS,1 if overlap else MHOURS):
        X.append(histories_arr[i:i+MHOURS,:])
        Y.append(histories_arr[i+MHOURS,:])
    X = np.array(X)
    Y = np.array(Y)
    print(f"Batched 'histories_arr' ({histories_arr.shape}) to 'X' ({X.shape}) and 'Y' ({Y.shape})")
    return X,Y


if __name__ == "__main__":
    RENEW_HISTORIES = True
    RENEW_MODEL = True
    SHOW_TRAIN_DS = False
    USE_RANDOM_FOREST = False
    MHOURS = 6
    TEST_SIZE = 0.2
    RECURSE_TO = 50
    MAX_TICKERS = 20
    EPOCHS = 1000
    BATCH_SIZE = 32
    PATIENCE = 25
    SHEET = "Stock"
    EXCHANGE = "NYQ"
    TICKER_COLUMN = "Ticker"
    TICKERS_TO_FOLLOW = json.load(open("TICKERS_TO_FOLLOW.json", "r"))

    
    histories_arr = None
    if not os.path.exists("histories_arr.npy") or RENEW_HISTORIES:
        if not TICKERS_TO_FOLLOW:
            tickers_from_excel = read_tickers_from_excel(sheet=SHEET, exchange=EXCHANGE, ticker_column=TICKER_COLUMN)
            with open("hand_picked_tickers.json", "r") as f:
                hand_picked = json.load(f)
            tickers_from_excel = list(set(tickers_from_excel + hand_picked))
            print(f"Tickers found from excel: {tickers_from_excel}")
        else:
            tickers_from_excel = TICKERS_TO_FOLLOW
        tickers = yf.tickers.Tickers(tickers_from_excel)
        print(f"Tickers found from yfinance: {tickers.tickers.keys()}")
        histories = get_histories(tickers, max_tickers=MAX_TICKERS)
        TICKERS_TO_FOLLOW = list(histories.keys())
        print(f"Selected tickers: {TICKERS_TO_FOLLOW}")
        histories_arr = histories_to_array(histories)
        with open("TICKERS_TO_FOLLOW.json", "w") as f:
            json.dump(TICKERS_TO_FOLLOW, f)
        np.save("histories_arr.npy", histories_arr)
    else:
        TICKERS_TO_FOLLOW = json.load(open("TICKERS_TO_FOLLOW.json", "r"))
        histories_arr = np.load("histories_arr.npy")

    IND_CONVERSION = {i:t for i,t in enumerate(TICKERS_TO_FOLLOW)}
    print(f"Shape of histories array: {histories_arr.shape}")

    # Batch X data into sequences of length MHOURS (from T to T+n), and Y data into sequences of length 1 (T+n+1)

    minmax_scaler = MinMaxScaler()
    # Fit the scaler to the first 1-TEST_SIZE of the data
    minmax_scaler = minmax_scaler.fit(histories_arr[:-int(histories_arr.shape[0]*TEST_SIZE),:])

    histories_arr = minmax_scaler.transform(histories_arr)

    X, Y = create_batch_XY(MHOURS, histories_arr)

    X_og = X.copy()
    Y_og = Y.copy()

    test_sz = int(X.shape[0]*TEST_SIZE)

    # Split the data into train and test sets
    X_train = X[:-test_sz,:,:]  
    Y_train = Y[:-test_sz,:]
    X_test = X[-test_sz:,:,:]
    Y_test = Y[-test_sz:,:]

    # Fit model, by showing it the data from the last MHOURS hours, and predicting the next hour
    if os.path.exists("model.h5") and not RENEW_MODEL:
        model = tf.keras.models.load_model("model.h5")
    else:
        model = create_tf_model(MHOURS, X.shape[2])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
        model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stop], shuffle=True)
        model.save("model.h5")
    # Predict the next hours for the test data
    Y_pred = model.predict(X_test)

    # Plot true values and predicted values for 5 selected stocks
    stock_idxs = np.random.randint(0,X.shape[2],5)
    fig, ax = plt.subplots()
    for stock_idx in stock_idxs:
        ax.plot(Y_test[:,stock_idx], label=f"True {IND_CONVERSION[stock_idx]}")
        ax.plot(Y_pred[:,stock_idx], label=f"Predicted {IND_CONVERSION[stock_idx]}")
    ax.legend()
    ax.set_title("True and predicted values for 5 stocks")
    
    # Test on the last TEST_SIZE of the data
    test_begin_idx = int(histories_arr.shape[0]*TEST_SIZE)
    X_overlap, Y_overlap = create_batch_XY(MHOURS, histories_arr[test_begin_idx-MHOURS:], overlap=True)
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
        ax.plot(Y_pred_overlap[:,stock_idx], label=f"Predicted {IND_CONVERSION[stock_idx]}")
    ax.legend()
    ax.set_title("True and predicted values for 5 stocks")

    # Calculate the error of the predictions
    mae_of_preds = np.mean(np.abs(Y_pred_overlap - Y_overlap), axis=0)
    print(f"Mean absolute error of predictions: {mae_of_preds}")

    # As a baseline, calculate how much the price changes per hour on the test data
    average_hourly_change = np.mean(np.abs(Y_overlap[1:] - Y_overlap[:-1]), axis=0)
    print(f"Average hourly change: {average_hourly_change}")

    plt.show()

