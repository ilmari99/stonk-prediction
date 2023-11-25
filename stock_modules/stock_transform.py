"""
Python module handling data reshaping and transformation.
"""

import numpy as np
import pandas as pd

def create_batch_xy(m_hours, histories_arr, y_updown=True, diff_data = True, overlap= False, output_scale = (0,1)):
    """
    Batch X data into sequences of length m_hours (from T to T+n), and Y data
    into sequences of length 1 (T+n+1)

    Args:
        m_hours (int): The number of hours to include in each input sequence.
        histories_arr (numpy.ndarray): A 2D array of historical stock prices,
        where each row represents a time step and each column represents a
        different stock.
        y_updown (bool, optional): Whether to predict the up/down movement.
            If True, the labels will be 1 if the price goes up, and 0 if it goes down.
            If False, the labels will be the actual price. Defaults to True.
    """
    x_matrix = []
    y_matrix = []
    if not isinstance(histories_arr, np.ndarray):
        histories_arr = np.array(histories_arr.values)
        print(histories_arr)
    for i in range(0,histories_arr.shape[0]-m_hours,1 if overlap else m_hours):
        x_matrix.append(histories_arr[i:i+m_hours,:])
        if not y_updown:
            y_matrix.append(histories_arr[i+m_hours,:])
        else:
            # If the price goes up, the label is 1, otherwise it is 0
            # If diff_data is True, the data is the difference, hence the label is 1 if the difference is positive, else 0
            past_prices = histories_arr[i+m_hours-1,:]
            curr_prices = histories_arr[i+m_hours,:]
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
            y_matrix.append(labels)
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
    """ Return a list of time differences between consecutive rows in the dataframe.
    The time difference at index i, is the amount of time between the rows i and i+1.
    The time difference is in hours.
    """
    time_deltas = []
    for i in range(len(df)-1):
        time_deltas.append((df[date_col_name].iloc[i+1] - df[date_col_name].iloc[i]).total_seconds()/3600)
    time_deltas.append(-1)
    df["Time Delta"] = time_deltas
    # datatype is int, since the measures are hourly
    df["Time Delta"] = df["Time Delta"].astype(int)
    cols = list(df.columns)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    df = df[cols]
    return df

