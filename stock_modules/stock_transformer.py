"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras import layers

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

def positional_encoding(length:PositiveInt, depth:PositiveInt):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class TokenEmbedding(layers.Layer):
    """
    Keras implementation of the Token embedding layer for Autoformers.
    """
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        # Create the 1D convolutional layer
        self.conv1d = layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            padding="circular",
            use_bias=False,
            kernel_initializer="he_normal",
            activation=layers.LeakyReLU
        )

    def call(self, inputs):
        # Transpose the input tensor so that the shape is
        # (batch_size, max_length, num_features)
        x = tf.transpose(inputs, perm=[0, 2, 1])

        # Apply the 1D convolutional layer
        x_embedded = self.conv1d(x)

        # Transpose the output tensor so that the shape is
        # (batch_size, max_length, d_model)
        x_embedded = tf.transpose(x_embedded, perm=[0, 2, 1])

        return x_embedded

class TemporalEmbedding(layers.Layer):
    """
    Keras implementation of the Temporal embedding layer for Autoformers.
    """
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

        # Create the embedding layers
        if self.embed_type == "fixed":
            # 10-hour day, only for weekdays
            self.hour_embed = layers.Embedding(10, d_model)
            self.weekday_embed = layers.Embedding(5, d_model)
            self.day_embed = layers.Embedding(32, d_model)
            self.month_embed = layers.Embedding(13, d_model)

            if self.freq == "t":
                self.minute_embed = layers.Embedding(4, d_model)
        else:
            self.time_embed = tf.keras.layers.Embedding(1000, d_model)

    def call(self, inputs):
        # Extract the time features from the input tensor
        hour = inputs[:, :, 3]
        weekday = inputs[:, :, 2]
        day = inputs[:, :, 1]
        month = inputs[:, :, 0]

        # Embed the time features
        if self.embed_type == "fixed":
            hour_embedded = self.hour_embed(hour)
            weekday_embedded = self.weekday_embed(weekday)
            day_embedded = self.day_embed(day)
            month_embedded = self.month_embed(month)

            if self.freq == "t":
                minute = inputs[:, :, 4]
                minute_embedded = self.minute_embed(minute)
            else:
                minute_embedded = tf.zeros_like(hour_embedded)

            x_embedded = \
                hour_embedded + weekday_embedded + day_embedded \
                    + month_embedded + minute_embedded
        else:
            x_embedded = self.time_embed(inputs)

        return x_embedded

class DataEmbedding(tf.keras.layers.Layer):
    """
    Keras implementation of the Positionless Data embedding layer for
    Autoformers.
    """
    def __init__(self, d_model, embed_type="fixed", freq="h", dropout_rate=0.1):
        super(DataEmbedding, self).__init__()

        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout_rate = dropout_rate

        # Create the embedding layers
        self.value_embedding = TokenEmbedding(d_model=self.d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model,
                                                    embed_type=self.embed_type,
                                                    freq=self.freq)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, inputs_mark):
        # Apply the value embedding to the input tensor
        x_embedded = self.value_embedding(inputs)

        # Apply the temporal embedding to the input tensor
        x_mark_embedded = self.temporal_embedding(inputs_mark)

        # Sum the two embeddings
        x_embedded = x_embedded + x_mark_embedded

        # Apply the dropout layer
        x_embedded = self.dropout(x_embedded)

        return x_embedded

def transformer_encoder(inputs:keras.Input,
                        head_size:PositiveInt,
                        num_heads:PositiveInt,
                        ff_dim:PositiveInt,
                        dropout:UnitFloat=0.):
    # Self-Attention and Add-Normalize
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
        )(inputs,inputs)
    res = layers.LayerNormalization(epsilon=1e-6, axis=2)(x+inputs)

    # Feed-Forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)

    # Add-Normalize
    return layers.LayerNormalization(epsilon=1e-6, axis=2)(x+res)

def transformer_decoder(inputs:keras.Input,
                        context:keras.Input,
                        head_size:PositiveInt,
                        num_heads:PositiveInt,
                        ff_dim:PositiveInt,
                        dropout:UnitFloat=0.):
    # Self-Attention and Add-Normalize
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
        )(inputs,inputs,use_causal_mask=True)
    res = layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+inputs)

    # Cross-Attention and Add-Normalize
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
        )(res,context)
    res = layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+res)

    # Feed-Forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.Dropout(dropout)(x)

    # Add-Normalize
    return layers.LayerNormalization(epsilon=1e-6, axis=(1,2))(x+res)

def build_model(input_shape:tuple[PositiveInt,PositiveInt],
                head_size:PositiveInt,
                num_heads:PositiveInt,
                ff_dim:PositiveInt,
                num_transformer_blocks:PositiveInt,
                mlp_units:tuple[PositiveInt,...],
                dropout:UnitFloat=0.,
                mlp_dropout:UnitFloat=0.):
    # Input declaration
    context = keras.Input(shape=input_shape)
    context_stamps = keras.Input(shape=(input_shape[0],1))
    inputs = keras.Input(shape=input_shape)
    input_stamps = keras.Input(shape=(input_shape[0],1))

    # Encoder layers
    x = DataEmbedding(head_size)(context, context_stamps)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Decoder layers
    y = DataEmbedding(inputs, input_stamps)
    for _ in range(num_transformer_blocks):
        y = transformer_decoder(y, x, head_size, num_heads, ff_dim, dropout)

    #Output layers
    y = layers.GlobalAveragePooling1D(data_format="channels_first")(y)
    for dim in mlp_units:
        y = layers.Dense(dim, activation="relu")(y)
        y = layers.Dropout(mlp_dropout)(y)
    outputs = layers.Dense(1, activation="linear")(y)

    return keras.Model([context,context_stamps,inputs,input_stamps],outputs)
