"""
Python module handling data reshaping and transformation.
"""

import numpy as np

def create_batch_xy(m_hours, histories_arr, overlap= False):
  """
  Batch X data into sequences of length m_hours (from T to T+n), and Y data
  into sequences of length 1 (T+n+1)

  Args:
    m_hours (int): The number of hours to include in each input sequence.
    histories_arr (numpy.ndarray): A 2D array of historical stock prices,
    where each row represents a time step and each column represents a
    different stock.
  """
  x_matrix = []
  y_matrix = []
  for i in range(0,histories_arr.shape[0]-m_hours,1 if overlap else m_hours):
    x_matrix.append(histories_arr[i:i+m_hours,:])
    y_matrix.append(histories_arr[i+m_hours,:])
  x_matrix = np.array(x_matrix)
  y_matrix = np.array(y_matrix)
  print(f"""
        Batched 'histories_arr' ({histories_arr.shape}) to 'X' ({x_matrix.shape})
        and 'Y' ({y_matrix.shape})
        """)
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
    dates = histories[ticker].index.values[-max_num_rows:]
    return np.array(values).T, dates
  return np.array(values).T
