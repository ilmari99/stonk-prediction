"""
Python module implementing the necessary layers for an
Autoformer in Keras
"""

import math

import tensorflow as tf
from tensorflow import keras
from keras import layers

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

@keras.saving.register_keras_serializable(package="Autoformer")
class Corr_Layer(layers.Layer):
    """
    Keras implementation of the autocorrelation attention
    mechanism
    """
    def __init__(self, k_factor:PositiveInt,
                 n_heads:PositiveInt,
                 d_keys = None,
                 d_values = None,
                 **kwargs):
        super(Corr_Layer, self).__init__(**kwargs)

        self.k_factor = k_factor
        self.n_heads = n_heads
        
        self.d_keys = d_keys
        self.d_values = d_values 

        self.query_proj = None
        self.key_proj = None
        self.value_proj = None
        self.out_proj = None

    def build(self, input_shape):
        self.d_keys = self.d_keys or input_shape[0][-1]//self.n_heads
        self.d_values = self.d_values or input_shape[0][-1]//self.n_heads

        self.query_proj = layers.Dense(self.d_keys*self.n_heads)
        self.key_proj = layers.Dense(self.d_keys*self.n_heads)
        self.value_proj = layers.Dense(self.d_values*self.n_heads)
        self.out_proj = layers.Dense(input_shape[0][-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k_factor": self.k_factor,
                "n_heads": self.n_heads,
                "d_keys": self.d_keys,
                "d_values": self.d_values,
                "query_proj": self.query_proj,
                "key_proj": self.key_proj,
                "value_proj": self.value_proj,
                "out_proj": self.out_proj
            }
        )

        return config

    def _time_delay_agg_full(self, values:tf.Tensor,
                             corr:tf.Tensor) -> tf.Tensor:
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Index init
        init_index = tf.repeat(
            tf.range(length)[tf.newaxis,tf.newaxis,tf.newaxis,:],
            repeats=[batch,head,channel,1])
        
        # Find top K
        top_k = int(self.k_factor * math.log(length))
        weights, delay = tf.math.top_k(corr, top_k)

        # Nonlinear weights for aggregation
        tmp_corr = layers.Softmax(axis=-1)(weights)

        # Aggregation
        tmp_values = tf.repeat(values, [1,1,1,2])
        delays_agg = tf.zeros_like(values, dtype=tf.float32)
        for idx in range(top_k):
            tmp_delay = init_index + tf.expand_dims(delay[..., idx], -1)
            pattern = tf.gather(tmp_values, tmp_delay, axis=-1)
            delays_agg += pattern * tf.expand_dims(tmp_corr[..., idx], -1)
        
        return delays_agg
    
    def call(self, inputs):
        queries, keys, values = inputs

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = tf.reshape(self.query_proj(queries), [B, L, H, -1])
        keys = tf.reshape(self.key_proj(keys), [B, S, H, -1])
        values = tf.reshape(self.value_proj(values), [B, S, H, -1])

        # Ensure dimension compatibility
        if L > S:
            zeros = tf.zeros_like(queries[:,:(L-S),:,:], dtype=tf.float32)
            keys = tf.concat([keys,zeros], axis=1)
            values = tf.concat([values,zeros], axis=1)
        else:
            keys = keys[:,:L,:,:]
            values = values[:,:L,:,:]

        # Period-based dependencies
        q_fft = tf.signal.rfft(tf.transpose(queries, [0,2,3,1]))
        k_fft = tf.signal.rfft(tf.transpose(keys, [0,2,3,1]))
        corr = tf.signal.irfft(q_fft*tf.math.conj(k_fft),
                               fft_length=L)
        
        out = tf.transpose(
            self._time_delay_agg_full(tf.transpose(values,[0,2,3,1]),
                                      corr),
            perm=[0,3,1,2]
        )
        out = tf.reshape(out, [B,L,-1])

        return self.out_proj(out)

@keras.saving.register_keras_serializable(package="Autoformer")
class Corr_Encoder(layers.Layer):
    """
    Keras implementation of the autoformer encoder
    """
    def __init__(self, **kwargs):
        super(Corr_Encoder, self).__init__(**kwargs)
        