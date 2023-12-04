"""
Python module handling figure generation.
"""
import matplotlib.pyplot as plt
import numpy as np
from invest_strategies import calculate_optimal_invest_strategy, calculate_profit_on_invest_strategy, strategy_mask_from_price_model, strategy_mask_from_updown_model
from stock_modules.stock_transform import create_batch_xy

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
            label = ind_conversion.get(col_id, "unk")
        else:
            label = ""
        ax.plot(arr[:,col_id], label=label)
    return ax

def plot_mask_and_data(mask, price_data, ind_conversion:dict={}):
    """ Plot the price data, and mark the buys and sells according to the mask.
    Make a subplot for each stock, with a maximum of 6 stocks.
    """
    assert mask.shape == price_data.shape, f"mask.shape = {mask.shape}, price_data.shape = {price_data.shape}"
    assert mask.shape[1] <= 6, f"mask.shape[1] = {mask.shape[1]}, but should be <= 6"
    fig, ax = plt.subplots(mask.shape[1], 1, sharex=True)
    for stock_idx in range(mask.shape[1]):
        ax[stock_idx].plot(price_data[:,stock_idx])
        buy_idx = np.argwhere(mask[:,stock_idx] == -1)
        sell_idx = np.argwhere(mask[:,stock_idx] == 1)
        ax[stock_idx].scatter(buy_idx, price_data[buy_idx,stock_idx], color="green", label="Buy")
        ax[stock_idx].scatter(sell_idx, price_data[sell_idx,stock_idx], color="red", label="Sell")
        profit = np.dot(mask[:,stock_idx], price_data[:,stock_idx])
        ax[stock_idx].set_title(f"Trading mask for {ind_conversion.get(stock_idx, 'unk')}, profit: {round(profit,2)} euros")
        ax[stock_idx].legend()
    fig.suptitle("True and predicted values for 6 stocks")
    plt.show()
    return fig, ax



def plot_strategy_based_on_predictions(data, transformed_data, model,
                                       window_hours, inversion= lambda x : x,
                                       ind_conversion:dict=None,
                                       model_type="updown",
                                       show = True,
                                       output_scale = (0,1)):
    """ Plot the true values of the data, and the predicted values.
    """

    _, y_test = create_batch_xy(window_hours, data, overlap=True)
    optimal_strat = calculate_optimal_invest_strategy(data)
    print(
            f"""
            Profit on optimal strategy:
            {calculate_profit_on_invest_strategy(data, optimal_strat)}
            """
        )

    invest_strat = strategy_mask_from_price_model(data, transformed_data, window_hours, model, inversion=inversion) if model_type == "price" else strategy_mask_from_updown_model(transformed_data, window_hours, model, output_scale=output_scale)
    print(f"Profit on invest strategy: {calculate_profit_on_invest_strategy(data, invest_strat)}")
    # Plot 6 columns from the test data, and mark predicted buys and sells
    fig, ax = plt.subplots(3,2)
    for i in range(3):
        for j in range(2):
            stock_idx = np.random.randint(0, data.shape[1])
            ax[i,j].plot(data[:,stock_idx], label=ind_conversion[stock_idx])
            # Mark sells with red dots, and buys with green dots
            sell_idx = np.argwhere(invest_strat[:,stock_idx] == 1)
            buy_idx = np.argwhere(invest_strat[:,stock_idx] == -1)
            ax[i,j].scatter(sell_idx, data[sell_idx,stock_idx], color="red", label="Sell")
            ax[i,j].scatter(buy_idx, data[buy_idx,stock_idx], color="green", label="Buy")
            stock_profit = np.dot(invest_strat[:,stock_idx], data[:,stock_idx])
            ax[i,j].set_title(f"Profit: {stock_profit}")
            ax[i,j].legend()
    fig.suptitle("True and predicted values for 6 stocks")
    if show:
        plt.show()
    return fig, ax
