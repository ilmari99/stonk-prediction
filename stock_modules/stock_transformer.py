"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from tensorflow import keras
import tensorflow as tf
from keras import layers

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
            self.day_embed = layers.Embedding(32, d_model)
            self.month_embed = layers.Embedding(13, d_model)

            if self.freq == "t":
                self.minute_embed = layers.Embedding(4, d_model)
        else:
            self.time_embed = layers.Embedding(1000, d_model)

    def call(self, inputs:keras.Input):
        # Extract the time features from the input tensor
        # hour = inputs[:, :, 3]
        hour = tf.slice(inputs, [0,0,3], [-1,-1,1])
        # weekday = inputs[:, :, 2]
        weekday = tf.slice(inputs, [0,0,2], [-1,-1,1])
        # day = inputs[:, :, 1]
        day = tf.slice(inputs, [0,0,1], [-1,-1,1])
        # month = inputs[:, :, 0]
        month = tf.slice(inputs, [0,0,0], [-1,-1,1])

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
                                                padding="same", use_bias=False,
                                                kernel_initializer="he_normal",
                                                activation=layers.LeakyReLU)
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model,
                                                    embed_type=self.embed_type,
                                                    freq=self.freq)
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs:keras.Input, inputs_mark:keras.Input):
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
                output_shape:PositiveInt,
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
    outputs = layers.Dense(output_shape, activation="linear")(y)

    return keras.Model([context,context_stamps,inputs,input_stamps],outputs)
