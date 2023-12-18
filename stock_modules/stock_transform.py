"""
Python module handling data reshaping and transformation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from datetime import datetime

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

def create_y_updown(past_prices, curr_prices,
                    diff_data = True, output_scale = (0,1)):
    """
    Create the labels for the up/down prediction model.
    If diff_data is True, the data is the difference,
    hence the label is 1 if the difference is positive, else 0.
    """
    if diff_data:
        labels = np.sign(curr_prices)
    else:
        labels = np.sign(curr_prices - past_prices)

    # If the output scale is (-1,1), then set 0s to -1
    if output_scale == (-1,1):
        labels[labels == 0] = -1
    # Else set -1 to 0
    elif output_scale == (0,1):
        labels[labels == -1] = 0
    return labels

def create_y_direction(past_prices, curr_prices,
                       threshold = 0.05, to_onehot = True):
    """ The label is 0 if the price goes down, 1 if it stays flat and
    2 if it goes up. The price stays the same, if the price changes less than
    threshold fraction of the past price.
    """
    labels = np.zeros_like(curr_prices)
    price_diffs = curr_prices - past_prices
    # The price has changed, if the absolute difference is greater than
    # thresh*past_price
    changed = np.abs(price_diffs) > threshold*past_prices
    # The price has gone up, if the difference is positive and the price has
    # changed
    up = (price_diffs > 0) & changed
    #print(f"Up: {up}")
    down = (price_diffs < 0) & changed
    #print(f"Down: {down}")
    same = ~changed
    #print(f"Same: {same}")
    labels[up] = 2
    labels[down] = 0
    labels[same] = 1
    #print(labels)
    if to_onehot:
        labels = tf.keras.utils.to_categorical(labels, num_classes=3)
    return labels


def create_batch_xy(m_hours, histories_arr, y_updown=False, y_direction=True,
                    overlap = False, output_scale = (0,1), to_onehot = True,
                    threshold = 0.02, create_labels = True):
    """
    Batch X data into sequences of length m_hours (from T to T+n), and Y data
    into sequences of length 1 (T+n+1)

    Args:
        m_hours (int): The number of hours to include in each input sequence.
        histories_arr (numpy.ndarray): A 2D array of historical stock prices,
        where each row represents a time step and each column represents a
        different stock.
        y_updown (bool, optional): Whether to predict the up/down movement.
            If True, the labels will be 1 if the price goes up,
            and 0 if it goes down.
            If False, the labels will be the actual price. Defaults to True.
    """
    # Data is differenced if any value in histories_arr is negative
    diff_data = np.any(histories_arr < 0)
    assert not (y_updown and y_direction), \
        "y_updown and y_direction cannot both be True."
    if create_labels:
        assert not (y_updown and y_direction), \
            "Either y_updown or y_direction must be True."
    #assert not create_labels and not (y_direction and diff_data), \
    # "If y direction is used, then the data must not be differenced."
    x_matrix = []
    y_matrix = []
    if not isinstance(histories_arr, np.ndarray):
        histories_arr = np.array( histories_arr.values )
        print(histories_arr)
    for i in range(0,histories_arr.shape[0]-m_hours,1 if overlap else m_hours):
        x_matrix.append(histories_arr[i:i+m_hours,:])
        past_prices = histories_arr[i+m_hours-1,:]
        curr_prices = histories_arr[i+m_hours,:]
        if not y_updown and not y_direction:
            y_matrix.append(histories_arr[i+m_hours,:])
        elif y_updown:
            # If the price goes up, the label is 1, otherwise it is 0
            # If diff_data is True, the data is the difference,
            # hence the label is 1 if the difference is positive, else 0
            labels = create_y_updown(past_prices, curr_prices,
                                     diff_data = diff_data,
                                     output_scale = output_scale)
            y_matrix.append(labels)
        elif y_direction:
            # We predict whether the price will go up, down or stay flat
            labels = create_y_direction(past_prices, curr_prices,
                                        threshold=threshold,
                                        to_onehot=to_onehot)
            y_matrix.append(labels)
    if y_direction and to_onehot and create_labels:

        # Then the y_matrix is a list of 2D arrays
        #  Convert it to numpy 3D array,
        # And check that it is of shape (num_samples, nstocks, 3)
        x_matrix = np.array(x_matrix)
        y_matrix = np.array(y_matrix)
        assert y_matrix.shape == (len(y_matrix), histories_arr.shape[1], 3), \
            f"""
            y_matrix.shape = {y_matrix.shape}, but should be ({len(y_matrix)},
            {histories_arr.shape[1]}, 3)"""
    else:
        x_matrix = np.array(x_matrix)
        y_matrix = np.array(y_matrix)
    print(
            f"""
            Batched 'histories_arr' ({histories_arr.shape}) to 'X'
            ({x_matrix.shape}) and 'Y' ({y_matrix.shape})
            """
        )
    return x_matrix,y_matrix

def histories_to_array(histories, get_dates=False):
    """
    Convert a dictionary of histories to a 2D array. If get_dates is True, also
    return the dates for the histories in a pandas DatetimeIndex.
    """
    values = []
    # Cutoff, so all histories have the same length
    max_num_rows = min([len(histories[ticker]) for ticker in histories])
    for ticker in histories:
        # take the last max_num_rows rows of the history
        stock_closes = histories[ticker]["Close"].values[-max_num_rows:]
        values.append(stock_closes)

    if get_dates:
        dates = histories[histories.keys()[-1]].index.values[-max_num_rows:]
        return np.array(values).T, dates

    return np.array(values).T

def add_time_delta_column(df : pd.DataFrame, date_col_name="date"):
    """
    Return a list of time differences between consecutive rows
    in the dataframe.
    The time difference at index i, is the amount of time between the rows
    i and i+1.
    The time difference is in hours.
    """
    time_deltas = []
    for i in range(len(df)-1):
        time_deltas.append((
                df[date_col_name].iloc[i+1] - df[date_col_name].iloc[i]
            ).total_seconds()/3600)
    time_deltas.append(-1)
    df["Time Delta"] = time_deltas
    # datatype is int, since the measures are hourly
    df["Time Delta"] = df["Time Delta"].astype(int)
    cols = list(df.columns)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    df = df[cols]
    return df

def create_transformer_onehot_xy(
        m_hours:PositiveInt, diff_data_np:np.ndarray,
        data_np:np.ndarray, timestamps_np:np.ndarray,
        static_thr:UnitFloat = 0.002,
        date_format:str = "%Y-%m-%d %H:%M:%S"
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    aux_diff_np = diff_data_np[m_hours:,:]
    aux_data_np = data_np[m_hours:,:]

    x = np.empty([diff_data_np.shape[0]-m_hours,
                    m_hours+1,
                    diff_data_np.shape[1]],
                  dtype=np.float32)
    x_ts = np.empty([x.shape[0], x.shape[1], 5],
                      dtype=np.int32)

    y = np.zeros([diff_data_np.shape[0]-m_hours,
                    diff_data_np.shape[1],
                    3],
                  dtype=np.float32)
    mask = np.zeros(y.shape[:2])

    for t in range(x.shape[0]):
        x[t,:,:] = diff_data_np[t:t+m_hours+1,:]

        aux_ts = [datetime.strptime(ts, date_format)
                  for ts in timestamps_np[t:t+m_hours+1]]
        x_ts[t,:,0] = [stamp.month-1 for stamp in aux_ts]
        x_ts[t,:,1] = [stamp.day-1 for stamp in aux_ts]
        x_ts[t,:,2] = [stamp.weekday() for stamp in aux_ts]
        x_ts[t,:,3] = [stamp.hour for stamp in aux_ts]
        x_ts[t,:,4] = [stamp.minute for stamp in aux_ts]

    # Class 0: Decreasing
    mask[aux_diff_np < -static_thr*aux_data_np] = 1
    y[:,:,0] = mask
    mask[:,:] = 0

    # Class 1: Static
    mask[np.abs(aux_diff_np) < static_thr*aux_data_np] = 1
    y[:,:,1] = mask
    mask[:,:] = 0

    # Class 2: Increasing
    mask[aux_diff_np > static_thr*aux_data_np] = 1
    y[:,:,2] = mask
    mask[:,:] = 0

    return (tf.convert_to_tensor(x, dtype=tf.float32),
            tf.convert_to_tensor(x_ts, dtype=tf.int32),
            tf.convert_to_tensor(y, dtype=tf.float32))
