"""
Python module handling model creation and training.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from stock_modules.stock_transformer import (Encoder,
                                             Decoder)
from stock_modules.stock_embed import DataEmbedding
from stock_modules.stock_autoformer import Autoformer

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

def create_direction_prediction_model(m, n):
    """ Create a model which predicts the direction of the stock price.
    The direction is up, down or flat for each stock,
    so the output is a (nx3) matrix.
    """
    inputs = layers.Input(shape=(m, n))
    #x = layers.BatchNormalization()(inputs)
    #x = layers.Bidirectional(
    x = layers.LSTM(200, return_sequences=True)(inputs)
    #)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(42, activation="relu")(x)
    # The output is a (nx3) matrix, where column is a one-hot
    # vector (probability distribution over the classes)
    x = layers.Dense(n*3, activation="relu")(x)
    outputs = layers.Reshape((3,n))(x)
    # Apply softmax to each column
    outputs = layers.Softmax(axis=1)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer="adam", loss="categorical_crossentropy",
    # metrics=["accuracy"])
    return model

def create_direction_prediction_model_rnn(m, n):
    """ Create a model which predicts the direction of the stock price.
    The direction is up, down or flat for each stock,
    so the output is a (nx3) matrix.
    """
    inputs = layers.Input(shape=(m, n))
    x = layers.BatchNormalization()(inputs)
    x = layers.SimpleRNN(64, return_sequences=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    # The output is a (nx3) matrix, where column is a one-hot
    # vector (probability distribution over the classes)
    x = layers.Dense(n*3, activation="relu")(x)
    outputs = layers.Reshape((3,n))(x)
    # Apply softmax to each column
    outputs = layers.Softmax(axis=1)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer="adam", loss="categorical_crossentropy",
    # metrics=["accuracy"])
    return model

def create_updown_prediction_model_dense(m, n, output_scale=(0, 1)):
    """ Dense model"""
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = layers.Input(shape=(m, n))

    # Normalize input data
    x = layers.BatchNormalization()(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(n,
                                   activation="tanh" if output_scale == (-1, 1)
                                   else "sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_updown_prediction_model(m, n, output_scale=(0, 1)):
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = layers.Input(shape=(m, n))

    # Normalize input data
    x = layers.BatchNormalization()(inputs)

    # Bidirectional LSTM layers with dropout
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                             kernel_regularizer=keras.regularizers.l2(0.01))
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                             kernel_regularizer=keras.regularizers.l2(0.01))
    )(x)
    x = layers.Dropout(0.2)(x)

    # Attention layer
    x = layers.Attention()([x, x, x])

    # Flatten and add dropout
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)

    # Dense layers
    x = layers.Dense(32, activation="relu")(x)

    # Output layer
    outputs = layers.Dense(n,
                activation="tanh" if output_scale == (-1, 1) else "sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_updown_prediction_model_rnn(m, n, output_scale=(0, 1)):
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = layers.Input(shape=(m, n))

    # Normalize input data
    x = layers.BatchNormalization()(inputs)

    # Bidirectional LSTM layers with dropout
    x = layers.SimpleRNN(64, return_sequences=True)(x)

    x = layers.Dropout(0.2)(x)

    x = layers.SimpleRNN(64)(x)

    x = layers.Dropout(0.2)(x)

    # Attention layer
    x = layers.Attention()([x, x, x])

    # Flatten and add dropout
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)

    # Dense layers
    x = layers.Dense(32, activation="relu")(x)

    # Output layer
    outputs = layers.Dense(
            n, activation="tanh" if output_scale == (-1, 1) else "sigmoid"
        )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def create_price_prediction_model(m, n):
    """
    Create an LSTM model which takes in m values of n stocks (m x n), and
    outputs the predicted value for each stock in the next time step (1 x n).
    m is the batch size, and n is the number of stocks.
    """
    inputs = layers.Input(shape=(m, n))
    #x = layers.BatchNormalization()(inputs)
    # Bidirectional LSTM layers
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer="l2")
        )(inputs)
    x = layers.Bidirectional(
        layers.LSTM(16, return_sequences=True)
        )(x)

    # Attention layer
    x = layers.Attention()([x, x, x])

    # Flatten and add dropout
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    # Dense layer
    x = layers.Dense(32, activation="relu")(x)

    # Output (1,N)
    outputs = layers.Dense(n, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")
    # Compile the model
    model.compile(optimizer="adam", loss="huber")
    return model

class SkipTDLoss(keras.losses.Loss):
    """ A loss funciton, that calculates the given loss but ignores the first
    column from y_true and y_pred.
    """
    def __init__(self, base_loss_fun, **kwargs):
        super().__init__(**kwargs)
        self.base_loss_fun = base_loss_fun

    def call(self, y_true, y_pred):
        # Skip first column
        return self.base_loss_fun(y_true[:,1:], y_pred[:,1:])

class SingleChannelMSE(keras.losses.Loss):
    def __init__(self, channel:PositiveInt = 0,
                 loss_fn = keras.losses.MeanSquaredError(),
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.channel = channel

    def call(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred[self.channel])

class MultiSoftmaxLoss(keras.losses.Loss):
    """
    Calculates the mean categorical crossentropy loss for each stock prediction.
    So input is (batch_size, nhours, nstocks), and output is
    (batch_size, 3, nstocks)
    The CCE/log loss is calculated for each stock and then averaged.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fun = keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        pred_losses = []
        # Calculate the softmax loss for each one-hot prediction
        # Y data is (3, nstocks), so take one column at a time, compare it to
        # the true value and calculate the loss
        for i in range(y_true.shape[2]):
            y_pred_i = y_pred[:,:,i]
            y_true_i = y_true[:,:,i]
            #print(f"Comparing y_true_i shape: {y_true_i.shape} to y_pred_i
            # shape: {y_pred_i.shape}")
            pred_losses.append(self.loss_fun(y_true_i, y_pred_i))
        # Now we have a list of losses for each stock, so we can calculate
        # the mean
        pred_losses = tf.stack(pred_losses, axis=0)
        mean_loss = tf.reduce_mean(pred_losses, axis=0)
        return mean_loss

class MultiAccuracy(keras.metrics.Metric):
    """ Calculates the average prediction accuracy for the predictions.
    So for stock, we calculate the accuracy of the prediction,
    sum them and then divide by the number of stocks.
    """
    def __init__(self, has_timedelta = False, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = 0 if not has_timedelta else 1
        self.accuracy = keras.metrics.CategoricalAccuracy()

    def update_state(self, y_true, y_pred, **kwargs):
        # Calculate the accuracy for each stock
        accs = []
        for i in range(self.start_idx, y_true.shape[2]):
            y_true_i = y_true[:,:,i]
            y_pred_i = y_pred[:,:,i]
            # y_pred_i to onehot
            y_pred_i = tf.one_hot(tf.argmax(y_pred_i, axis=1), depth=3)
            accs.append(self.accuracy(y_true_i, y_pred_i))
        # Now we have a list of accuracies for each stock, so we can calculate
        # the mean
        accs = tf.stack(accs, axis=0)
        mean_acc = tf.reduce_mean(accs, axis=0)
        self.mean_acc = mean_acc

    def result(self):
        return self.mean_acc

    def reset_state(self):
        pass

def create_transformer_model(m:PositiveInt, n:PositiveInt,
                output_dim:PositiveInt=3,
                head_size:PositiveInt=16,
                num_heads:PositiveInt=16,
                ff_dim:PositiveInt=32,
                num_transformer_blocks:PositiveInt=1,
                mlp_units:tuple[PositiveInt,...]=(64,),
                dropout:UnitFloat=0.01,
                mlp_dropout:UnitFloat=0.,
                class_first=False):
    input_shape = (m,n)

    # Input declaration -> [BxLxN, BxLx1]
    context = keras.Input(shape=input_shape)
    context_ts = keras.Input(shape=(m,4))
    inputs = keras.Input(shape=input_shape)
    input_ts = keras.Input(shape=(m,4))

    # Encoder layers -> BxLxC
    x = DataEmbedding(head_size)([context, context_ts])
    for _ in range(num_transformer_blocks):
        x = Encoder(head_size, num_heads, ff_dim, dropout)(x)

    # Decoder layers -> BxLxC
    y = DataEmbedding(head_size)([inputs, input_ts])
    for _ in range(num_transformer_blocks):
        y = Decoder(head_size, num_heads, ff_dim, dropout)([y,x])

    # Output layers -> BxNx3
    y = layers.Conv1DTranspose(filters=n, kernel_size=3, padding="same")(y)
    y = layers.Permute((2,1))(y)
    for dim in mlp_units:
        y = layers.Dense(dim, activation="relu")(y)
        y = layers.Dropout(mlp_dropout)(y)
    outputs = layers.Dense(output_dim, activation="linear")(y)

    if class_first:
        outputs = layers.Permute((2,1))(outputs)

    return keras.Model([context,context_ts,inputs,input_ts],outputs)

def create_autoformer_model(m:PositiveInt, n:PositiveInt, **config):
    x_enc_shape = (m,n)
    x_enc_marks_shape = (m,5)
    x_dec_marks_shape = (config["O"], 5)

    x_enc = keras.Input(shape=x_enc_shape)
    x_enc_marks = keras.Input(shape=x_enc_marks_shape)
    x_dec_marks = keras.Input(shape=x_dec_marks_shape)

    y_dec_classes = Autoformer(**config)([x_enc,x_enc_marks,x_dec_marks])

    return keras.Model([x_enc,x_enc_marks,x_dec_marks], y_dec_classes)
