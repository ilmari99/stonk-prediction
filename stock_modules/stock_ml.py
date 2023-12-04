"""
Python module handling model creation and training.
"""

from tensorflow import keras
from keras import layers

from stock_modules.stock_transformer import (DataEmbedding,
                                             transformer_encoder,
                                             transformer_decoder)

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

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

def create_transformer_model(m:PositiveInt, n:PositiveInt,
                output_dim:PositiveInt=3,
                head_size:PositiveInt=16,
                num_heads:PositiveInt=16,
                ff_dim:PositiveInt=32,
                num_transformer_blocks:PositiveInt=1,
                mlp_units:tuple[PositiveInt,...]=(64,),
                dropout:UnitFloat=0.01,
                mlp_dropout:UnitFloat=0.):
    input_shape = (m,n)

    # Input declaration -> [BxLxN, BxLx1]
    context = keras.Input(shape=input_shape)
    context_stamps = keras.Input(shape=(m,1))
    inputs = keras.Input(shape=input_shape)
    input_stamps = keras.Input(shape=(m,1))

    # Encoder layers -> BxLxC
    x = DataEmbedding(head_size)(context, context_stamps)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Decoder layers -> BxLxC
    y = DataEmbedding(head_size)(inputs, input_stamps)
    for _ in range(num_transformer_blocks):
        y = transformer_decoder(y, x, head_size, num_heads, ff_dim, dropout)

    # Output layers -> BxNx3
    y = layers.Conv1DTranspose(filters=n, kernel_size=3, padding="same")(y)
    y = layers.Permute((2,1))(y)
    for dim in mlp_units:
        y = layers.Dense(dim, activation="relu")(y)
        y = layers.Dropout(mlp_dropout)(y)
    outputs = layers.Dense(output_dim, activation="linear")(y)

    return keras.Model([context,context_stamps,inputs,input_stamps],outputs)
