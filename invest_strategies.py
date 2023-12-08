""" This file contains functions for creating a hypothetical trading strategy
based on a model that predicts the price of stocks one hour ahead.
Also possible to create a hypothetically optimal trading strategy, that assumes
price is known one hour ahead.
"""

from tensorflow import keras
import numpy as np
from stock_modules.stock_transform import (create_batch_xy,
                                           create_transformer_onehot_xy)

def calculate_optimal_invest_strategy(data : np.ndarray) -> np.ndarray:
    """ Calculates the optimaltrading mask for the data. This assumes that we
    know 1h in to the future.
    The strategy:
    If we are not holding, and price at T+1 is higher than price at T, we buy 1
    (marked as -1)
    If we are holding, and price at T+1 is lower than price at T, we sell 1
    (marked as 1)
    """

    mask = np.zeros(data.shape)
    for stock_idx in range(data.shape[1]):
        stock_data = data[:,stock_idx]
        holding_stock = False
        for i in range(stock_data.shape[0]):
            # If we are not holding the stock, buy if the price is the lowest
            # so far
            if not holding_stock and stock_data[i] == np.min(stock_data[i:]):
                mask[i,stock_idx] = -1
                holding_stock = True
                # If we are holding the stock, sell if the price is the highest
                # so far
            elif holding_stock and stock_data[i] == np.max(stock_data[i:]):
                mask[i,stock_idx] = 1
                holding_stock = False
            if holding_stock:
                mask[-1,stock_idx] = 1
    return mask

def calculate_profit_on_invest_strategy(data : np.ndarray,
                                        mask : np.ndarray) -> float:
    """ Calculate how much profit would be made by using the mask to buy and
    sell stocks.
    For an individual stock, the profit is the dot product of the mask and the
    stock price (assuming mask is correct).
    """
    profit = 0
    for stock_idx in range(data.shape[1]):
        profit += np.dot(mask[:,stock_idx], data[:,stock_idx])
    return profit

def strategy_mask_from_direction_model(transformed_data:np.ndarray,
                                       window_hours:int,
                                       model:keras.Model,
                                       is_transformer:bool = False,
                                       original_data:np.ndarray = None,
                                       time_stamps:np.ndarray = None
                                    ) -> np.ndarray:
    """
    Calculates a trading mask based on a model that predicts whether the
    price of the stock will go up, down or stay flat one hour ahead.
    """

    mask = np.zeros(transformed_data.shape)

    if is_transformer:
        x, x_ts, _ = create_transformer_onehot_xy(window_hours,
                                                  transformed_data,
                                                  original_data,
                                                  time_stamps)
        y_pred = model.predict([x, x_ts, x, x_ts])
    else:
        x, _ = create_batch_xy(window_hours, transformed_data,
                           overlap=True, y_direction=True,
                           to_onehot=True, create_labels=False)
        y_pred = model.predict(x)

    holding_bool = np.zeros(transformed_data.shape[1])
    # Loop through batches, and make the mask based on the predictions
    for i in range(x.shape[0]):
        ith_predictions = y_pred[i,:,:]
        # Calculate the argmax for eacl column
        ith_predictions = np.argmax(ith_predictions, axis=0)
        #print(ith_predictions)
        # Check all stocks
        for stock_idx in range(transformed_data.shape[1]):
            # If we are not holding the stock,
            # buy if the price is predicted to go up
            if not holding_bool[stock_idx] and ith_predictions[stock_idx] == 2:
                mask[i,stock_idx] = -1
                holding_bool[stock_idx] = 1
            # If we are holding the stock,
            # sell if the price is predicted to go down
            elif holding_bool[stock_idx] and ith_predictions[stock_idx] == 0:
                mask[i,stock_idx] = 1
                holding_bool[stock_idx] = 0
    # Sell all remaining stocks at the end if we have any
    for stock_idx in range(transformed_data.shape[1]):
        if holding_bool[stock_idx]:
            mask[-1,stock_idx] = 1
    return mask


def strategy_mask_from_updown_model(transformed_data : np.ndarray,
                                    window_hours : int,
                                    model,
                                    output_scale = (0,1),
                                    ) -> np.ndarray:
    """
    Calculates a trading mask based on a model that predicts whether the
    price of the stock will go up or down one hour ahead.

    Args:
        data (np.ndarray): The original data of the stock prices.
        transformed_data (np.ndarray): The transformed data used for prediction.
        window_hours (int): The number of hours in the window for prediction.
        model: The model used for price prediction.
        output_scale (tuple, optional): The scale of the output of the model.
        If the prediction is over (b - a) / 2,
        the price is predicted to go up, else down. Defaults to (0,1, sigmoid).

    Returns:
        np.ndarray: The trading mask indicating whether to buy, sell,
        or hold stocks.
    """

    assert output_scale[0] < output_scale[1], \
        "The first element of output_scale must be smaller than the second"
    assert hasattr(model, "predict"), "The model must have a predict method"
    mask = np.zeros(transformed_data.shape)
    xt, _ = create_batch_xy(window_hours, transformed_data, overlap=True)
    x, _ = create_batch_xy(window_hours, transformed_data, overlap=True)
    # For each hour, predict whether the price will go up or down
    y_pred = model.predict(xt)
    y_pred = np.squeeze(y_pred)
    # make sure the outputs are in the range [0,1]
    y_pred = np.clip(y_pred, output_scale[0], output_scale[1])
    holding_bool = np.zeros(transformed_data.shape[1])
    # Loop through batches, and make the mask based on the predictions
    for i in range(x.shape[0]):
        ith_predictions = y_pred[i,:]
        going_up = ith_predictions > (output_scale[1] - output_scale[0]) / 2
        for stock_idx in range(transformed_data.shape[1]):
            # If we are not holding the stock,
            # buy if the price is predicted to go up
            if not holding_bool[stock_idx] and going_up[stock_idx]:
                mask[i,stock_idx] = -1
                holding_bool[stock_idx] = 1
            # If we are holding the stock,
            # sell if the price is predicted to go down
            elif holding_bool[stock_idx] and not going_up[stock_idx]:
                mask[i,stock_idx] = 1
                holding_bool[stock_idx] = 0
    # Sell all remaining stocks at the end if we have any
    for stock_idx in range(transformed_data.shape[1]):
        if holding_bool[stock_idx]:
            mask[-1,stock_idx] = 1
    return mask


def strategy_mask_from_price_model(data : np.ndarray,
                                   transformed_data : np.ndarray,
                                   window_hours : int,
                                   model,
                                   inversion = lambda x : x
                                   ) -> np.ndarray:
    """
    Calculates a trading mask based on a model that predicts
    the price of the stock one hour ahead.

    Args:
        data (np.ndarray): The original data of the stock prices.
        transformed_data (np.ndarray): The transformed data used for prediction.
        window_hours (int): The number of hours in the window for prediction.
        model: The model used for price prediction.
        inversion (function, optional): The function used to invert
        the predicted prices. Defaults to lambda x: x.

    Returns:
        np.ndarray: The trading mask indicating whether to buy, sell,
        or hold stocks.
    """

    mask = np.zeros(data.shape)
    xt, _ = create_batch_xy(window_hours, transformed_data, overlap=True)
    x, _ = create_batch_xy(window_hours, data, overlap=True)
    # Predict the next hours based on the _transformed data_, and the inversion
    y_pred = model.predict(xt)
    y_pred = inversion(y_pred)

    # We are not holding any stocks at the beginning
    holding_bool = np.zeros(data.shape[1])
    # Loop through the last hour of X and the model predictions
    for i in range(x.shape[0]):
        current_prices = x[i,-1,:]
        predicted_prices = y_pred[i,:]
        # Check all stocks
        for stock_idx in range(data.shape[1]):
        # if we are not holding a stock and price at T+1 is higher
        # than price at T, buy
            if (holding_bool[stock_idx] == 0
                    and predicted_prices[stock_idx] >
                        current_prices[stock_idx]):
                mask[i,stock_idx] = -1
                holding_bool[stock_idx] = 1

            # if we are holding a stock and price at T+1 is lower than
            # price at T, sell
            elif (holding_bool[stock_idx] == 1
                    and predicted_prices[stock_idx]
                        < current_prices[stock_idx]):
                mask[i,stock_idx] = 1
                holding_bool[stock_idx] = 0

    # Sell all remaining stocks at the end if we have any
    for stock_idx in range(data.shape[1]):
        if holding_bool[stock_idx] == 1:
            mask[-1,stock_idx] = 1
    return mask

