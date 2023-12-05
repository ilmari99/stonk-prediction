"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from tensorflow import keras
import tensorflow as tf
from keras import layers

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

class TemporalEmbedding(layers.Layer):
    """
    Keras implementation of the Temporal embedding layer for Autoformers.
    """
    def __init__(self, d_model:PositiveInt,
                 embed_type:str="fixed",
                 freq:str="h"):
        super(TemporalEmbedding, self).__init__()

        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

        # Create the embedding layers
        if self.embed_type == "fixed":
            # 10-hour day, only for weekdays
            self.hour_embed = layers.Embedding(10, d_model)
            self.weekday_embed = layers.Embedding(5, d_model)
            self.day_embed = layers.Embedding(31, d_model)
            self.month_embed = layers.Embedding(12, d_model)

            if self.freq == "t":
                self.minute_embed = layers.Embedding(4, d_model)
        else:
            self.time_embed = layers.Embedding(1000, d_model)

    def call(self, inputs:keras.Input):
        # Extract the time features from the input tensor
        # month = inputs[:, :, 0]
        month = layers.Reshape((-1,))(tf.slice(inputs, [0,0,0], [-1,-1,1]))
        # day = inputs[:, :, 1]
        day = layers.Reshape((-1,))(tf.slice(inputs, [0,0,1], [-1,-1,1]))
        # weekday = inputs[:, :, 2]
        weekday = layers.Reshape((-1,))(tf.slice(inputs, [0,0,2], [-1,-1,1]))
        # hour = inputs[:, :, 3]
        hour = layers.Reshape((-1,))(tf.slice(inputs, [0,0,3], [-1,-1,1]))

        # Embed the time features
        if self.embed_type == "fixed":
            hour_embedded = self.hour_embed(hour)
            weekday_embedded = self.weekday_embed(weekday)
            day_embedded = self.day_embed(day)
            month_embedded = self.month_embed(month)

            if self.freq == "t":
                # minute = inputs[:, :, 4]
                minute = tf.slice(inputs, [0,0,4], [-1,-1,1])
                minute_embedded = self.minute_embed(minute)
            else:
                minute_embedded = tf.zeros_like(hour_embedded)

            x_embedded = \
                hour_embedded + weekday_embedded + day_embedded \
                    + month_embedded + minute_embedded
        else:
            x_embedded = self.time_embed(inputs)

        return x_embedded

class DataEmbedding(layers.Layer):
    """
    Keras implementation of the Positionless Data embedding layer for
    Autoformers.
    """
    def __init__(self, d_model:PositiveInt,
                 embed_type:str="fixed", freq:str="h",
                 dropout_rate:UnitFloat=0.1):
        super(DataEmbedding, self).__init__()

        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        self.dropout_rate = dropout_rate

        # Create the embedding layers
        self.value_embedding = layers.Conv1D(filters=d_model, kernel_size=3,
                                                padding="same",
                                                kernel_initializer="he_normal",
                                                activation="leaky_relu")
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model,
                                                    embed_type=self.embed_type,
                                                    freq=self.freq)
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs:keras.Input):
        x, x_ts = inputs

        # Apply the value embedding to the input tensor
        x_embedded = self.value_embedding(x)

        # Apply the temporal embedding to the input tensor
        x_ts_embedded = self.temporal_embedding(x_ts)

        # Sum the two embeddings
        x_embedded = x_embedded + x_ts_embedded

        # Apply the dropout layer
        x_embedded = self.dropout(x_embedded)

        return x_embedded

class Encoder(layers.Layer):
    """
    Keras implementation of the canonical transformer encoder (optimized
    for time-series)
    """
    def __init__(self, head_size:PositiveInt,
                 num_heads:PositiveInt, ff_dim:PositiveInt,
                 dropout:UnitFloat=0.):
        super(Encoder, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.self_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.ff_dim,
                              kernel_size=1,
                              activation="relu"),
                layers.Conv1D(filters=input_shape[-1], kernel_size=1),
                layers.Dropout(rate=self.dropout)
            ]
        )

    def call(self, inputs):
        x = self.self_attention(inputs,inputs)
        res = self.norm(x+inputs)

        x = self.ff_layer(res)
        x = self.norm(x+res)

        return x

class Decoder(layers.Layer):
    """
    Keras implementation of the canonical transformer decoder (optimized
    for time-series)
    """
    def __init__(self, head_size:PositiveInt,
                 num_heads:PositiveInt, ff_dim:PositiveInt,
                 dropout:UnitFloat=0.):
        super(Decoder, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.self_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout)
        self.cross_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.ff_dim,
                              kernel_size=1,
                              activation="relu"),
                layers.Conv1D(filters=input_shape[0][-1], kernel_size=1),
                layers.Dropout(rate=self.dropout)
            ]
        )

    def call(self, inputs):
        target, context = inputs

        x = self.self_attention(target,target,use_causal_mask=True)
        res = self.norm(x+target)

        x = self.cross_attention(res,context)
        res = self.norm(x+res)

        x = self.ff_layer(res)
        x = self.norm(x+res)

        return x

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
