"""
Python module handling figure generation.
"""
import matplotlib.pyplot as plt

def plot_numpy_arr_cols(arr, ax=None, ind_conversion:dict=None):
  """ Plot the columns of a 2d numpy array.
  If ax is None, create a new figure.
  ind_conversion is a dictionary mapping column indices to names, which will
  be used as labels.
  """
  if ax is None:
    _, ax = plt.subplots()
  for col_id in range(arr.shape[1]):
    if col_id in ind_conversion:
      label = ind_conversion[col_id]
    else:
      label = ""
      ax.plot(arr[:,col_id], label=label)
  return ax
