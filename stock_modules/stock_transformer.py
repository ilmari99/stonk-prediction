"""
Python module implementing the necessary layers for a classic
Transformer in Keras
"""

from tensorflow import keras
from keras import layers

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

@keras.saving.register_keras_serializable(package="Transformer")
class Encoder(layers.Layer):
    """
    Keras implementation of the canonical transformer encoder (optimized
    for time-series)
    """
    def __init__(self, head_size:PositiveInt,
                 num_heads:PositiveInt, ff_dim:PositiveInt,
                 dropout_rate:UnitFloat=0.,
                 **kwargs):
        super(Encoder, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.self_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout_rate)
        self.norm_self = layers.LayerNormalization(epsilon=1e-6)

        self.ff_layer = None
        self.norm_ff = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.ff_dim,
                              kernel_size=1,
                              activation="relu"),
                layers.Conv1D(filters=input_shape[-1], kernel_size=1),
                layers.Dropout(rate=self.dropout_rate)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_size": self.head_size,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "self_attention": self.self_attention,
                "norm_self": self.norm_self,
                "ff_layer": self.ff_layer,
                "norm_ff": self.norm_ff
            }
        )

        return config

    def call(self, inputs):
        x = self.self_attention(inputs,inputs)
        res = self.norm_self(x+inputs)

        x = self.ff_layer(res)
        x = self.norm_ff(x+res)

        return x

@keras.saving.register_keras_serializable(package="Transformer")
class Decoder(layers.Layer):
    """
    Keras implementation of the canonical transformer decoder (optimized
    for time-series)
    """
    def __init__(self, head_size:PositiveInt,
                 num_heads:PositiveInt, ff_dim:PositiveInt,
                 dropout_rate:UnitFloat=0.,
                 **kwargs):
        super(Decoder, self).__init__()

        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.self_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout_rate)
        self.norm_self = layers.LayerNormalization(epsilon=1e-6)

        self.cross_attention = layers.MultiHeadAttention(key_dim=self.head_size,
                                             num_heads=self.num_heads,
                                             dropout=self.dropout_rate)
        self.norm_cross = layers.LayerNormalization(epsilon=1e-6)

        self.ff_layer = None
        self.norm_ff = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.ff_dim,
                              kernel_size=1,
                              activation="relu"),
                layers.Conv1D(filters=input_shape[0][-1], kernel_size=1),
                layers.Dropout(rate=self.dropout_rate)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_size": self.head_size,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "self_attention": self.self_attention,
                "norm_self": self.norm_self,
                "cross_attention": self.cross_attention,
                "norm_cross": self.norm_cross,
                "ff_layer": self.ff_layer,
                "norm_ff": self.norm_ff
            }
        )

        return config

    def call(self, inputs):
        target, context = inputs

        x = self.self_attention(target,target,use_causal_mask=True)
        res = self.norm_self(x+target)

        x = self.cross_attention(res,context)
        res = self.norm_cross(x+res)

        x = self.ff_layer(res)
        x = self.norm_ff(x+res)

        return x
