"""
Python module implementing the necessary layers for an
Autoformer in Keras
"""

import math

import tensorflow as tf
from tensorflow import keras
from keras import layers

from stock_modules.stock_embed import DataEmbedding

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

UnitFloat = Annotated[float, Field(strict=True, ge=0., le=1.)]

@keras.saving.register_keras_serializable(package="Autoformer")
class CorrLayer(layers.Layer):
    """
    Keras implementation of the autocorrelation attention
    mechanism
    """
    def __init__(self, k_factor:PositiveInt,
                 n_heads:PositiveInt,
                 d_keys = None,
                 d_values = None,
                 dropout_rate:UnitFloat = 0.,
                 **kwargs):
        super(CorrLayer, self).__init__(**kwargs)

        self.k_factor = k_factor
        self.n_heads = n_heads

        self.d_keys = d_keys
        self.d_values = d_values

        self.dropout = layers.Dropout(dropout_rate)

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
        length = values.shape[-1]

        # Index init
        init_index = tf.tile(
            tf.range(length)[tf.newaxis,tf.newaxis,tf.newaxis,:],
            tf.concat([tf.shape(values)[:3], tf.constant([1], dtype=tf.int32)],
                      axis=-1)
        )

        # Find top K
        top_k = int(self.k_factor * math.log(length))
        weights, delay = tf.math.top_k(corr, top_k)

        # Nonlinear weights for aggregation
        tmp_corr = layers.Softmax(axis=-1)(weights)

        # Aggregation
        tmp_values = tf.tile(values, [1,1,1,2])
        delays_agg = tf.zeros_like(values, dtype=tf.float32)
        for idx in range(top_k):
            tmp_delay = init_index + tf.expand_dims(delay[..., idx], -1)
            pattern = tf.gather(tmp_values, tmp_delay, axis=-1, batch_dims=3)
            delays_agg += pattern * tf.expand_dims(tmp_corr[..., idx], -1)

        return delays_agg

    def call(self, inputs):
        queries, keys, values = inputs

        _, q_len, d_out = queries.shape
        _, k_len, _ = keys.shape
        n_heads = self.n_heads

        queries = self.query_proj(queries)
        queries = tf.reshape(queries,
                             [-1, q_len, n_heads, self.d_keys*self.n_heads])
        keys = tf.reshape(self.key_proj(keys),
                          [-1, k_len, n_heads, self.d_keys*self.n_heads])
        values = tf.reshape(self.value_proj(values),
                            [-1, k_len, n_heads, self.d_values*self.n_heads])

        # Ensure dimension compatibility
        if q_len > k_len:
            zeros = tf.zeros_like(queries[:,:(q_len-k_len),:,:],
                                  dtype=tf.float32)
            keys = tf.concat([keys,zeros], axis=1)
            values = tf.concat([values,zeros], axis=1)
        else:
            keys = keys[:,:q_len,:,:]
            values = values[:,:q_len,:,:]

        # Period-based dependencies
        q_fft = tf.signal.rfft(tf.transpose(queries, [0,2,3,1]))
        k_fft = tf.signal.rfft(tf.transpose(keys, [0,2,3,1]))
        corr = tf.signal.irfft(q_fft*tf.math.conj(k_fft),
                               fft_length=[q_len])

        out = tf.transpose(
            self._time_delay_agg_full(tf.transpose(values,[0,2,3,1]),
                                      corr),
            perm=[0,3,1,2]
        )
        out = self.out_proj(tf.reshape(out, [-1,q_len,d_out]))

        return self.dropout(out)

@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoderLayer(layers.Layer):
    """
    Keras implementation of a single Autoformer encoder layer
    """
    def __init__(self, k_factor:PositiveInt = 1,
                 n_heads:PositiveInt = 1,
                 d_ff:PositiveInt=None,
                 moving_avg:PositiveInt=25,
                 dropout_rate:UnitFloat=0.1,
                 activation="relu", **kwargs):
        super(CorrEncoderLayer, self).__init__(**kwargs)

        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.moving_avg = layers.AvgPool1D(pool_size=moving_avg, strides=1,
                                           padding="same")
        self.attn_layer = CorrLayer(k_factor=k_factor,
                                    n_heads=n_heads,
                                    dropout_rate=dropout_rate)

        self.ff_layer = None

    def build(self, input_shape):
        self.d_ff = self.d_ff or 4*input_shape[-1]
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.d_ff, kernel_size=1,
                              use_bias=False,
                              activation=self.activation),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(filters=input_shape[-1], kernel_size=1,
                              use_bias=False, activation=None),
                layers.Dropout(self.dropout_rate)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_ff": self.d_ff,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "moving_avg": self.moving_avg,
                "attn_layer": self.attn_layer,
                "ff_layer": self.ff_layer
            }
        )
        return config

    def _series_decomp(self, inputs):
        trend = self.moving_avg(inputs)
        seasonality = inputs - trend
        return seasonality, trend

    def call(self, inputs):
        x = inputs + self.attn_layer([inputs,inputs,inputs])
        x, _ = self._series_decomp(x)

        y = self.ff_layer(x)
        y, _ = self._series_decomp(x+y)

        return y

@keras.saving.register_keras_serializable(package="Autoformer")
class CorrEncoder(layers.Layer):
    """
    Keras implementation of the Autoformer encoder block
    """
    def __init__(self,
                 n_layers:PositiveInt,
                 k_factor:PositiveInt = 1,
                 n_heads:PositiveInt = 1,
                 d_ff:PositiveInt=None,
                 moving_avg:PositiveInt=25,
                 dropout_rate:UnitFloat=0.1,
                 activation="relu",
                 **kwargs):
        super(CorrEncoder, self).__init__(**kwargs)
        self.enc_layers = [
            CorrEncoderLayer(k_factor=k_factor,
                             n_heads=n_heads,
                             d_ff=d_ff,
                             moving_avg=moving_avg,
                             dropout_rate=dropout_rate,
                             activation=activation)
                for _ in range(n_layers)
        ]
        self.norm_layer = layers.Normalization([-2,-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dec_layers": self.enc_layers,
                "norm_layer": self.norm_layer
            }
        )

    def call(self, inputs):
        x = inputs
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        return self.norm_layer(x)

@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoderLayer(layers.Layer):
    """
    Keras implementation of a single Autoformer decoder layer
    """
    def __init__(self, d_out:PositiveInt,
                 k_factor:PositiveInt = 1,
                 n_heads:PositiveInt = 1,
                 d_ff:PositiveInt=None,
                 moving_avg:PositiveInt=25,
                 dropout_rate:UnitFloat=0.1,
                 activation="relu",
                 **kwargs):
        super(CorrDecoderLayer, self).__init__(**kwargs)

        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.moving_avg = layers.AvgPool1D(pool_size=moving_avg, strides=1,
                                           padding="same")
        self.self_attn = CorrLayer(k_factor=k_factor,
                                    n_heads=n_heads,
                                    dropout_rate=dropout_rate)
        self.cross_attn = CorrLayer(k_factor=k_factor,
                                    n_heads=n_heads,
                                    dropout_rate=dropout_rate)

        self.ff_layer = None

        self.out_proj = layers.Conv1D(filters=d_out, kernel_size=3,
                                      strides=1, padding="same",
                                      use_bias=False)

    def build(self, input_shape):
        self.d_ff = self.d_ff or 4*input_shape[0][-1]
        self.ff_layer = keras.Sequential(
            [
                layers.Conv1D(filters=self.d_ff, kernel_size=1,
                              use_bias=False,
                              activation=self.activation),
                layers.Dropout(self.dropout_rate),
                layers.Conv1D(filters=input_shape[0][-1], kernel_size=1,
                              use_bias=False, activation=None),
                layers.Dropout(self.dropout_rate)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_ff": self.d_ff,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "moving_avg": self.moving_avg,
                "self_attn": self.self_attn,
                "cross_attn": self.cross_attn,
                "ff_layer": self.ff_layer
            }
        )
        return config

    def _series_decomp(self, inputs):
        trend = self.moving_avg(inputs)
        seasonality = inputs - trend
        return seasonality, trend

    def call(self, inputs):
        x, cross = inputs
        x = x + self.self_attn([x,x,x])
        x, xt_1 = self._series_decomp(x)

        x = x + self.cross_attn([x,cross,cross])
        x, xt_2 = self._series_decomp(x)

        y = self.ff_layer(x)
        y, xt_3 = self._series_decomp(x+y)

        residual_trend = self.out_proj(xt_1+xt_2+xt_3)

        return y, residual_trend

@keras.saving.register_keras_serializable(package="Autoformer")
class CorrDecoder(layers.Layer):
    """
    Keras implementation of the Autoformer decoder block
    """
    def __init__(self,
                 d_out:PositiveInt,
                 n_layers:PositiveInt,
                 k_factor:PositiveInt = 1,
                 n_heads:PositiveInt = 1,
                 d_ff:PositiveInt=None,
                 moving_avg:PositiveInt=25,
                 dropout_rate:UnitFloat=0.1,
                 activation="relu",
                 **kwargs):
        super(CorrDecoder, self).__init__(**kwargs)
        self.dec_layers = [
            CorrDecoderLayer(d_out=d_out,
                             k_factor=k_factor,
                             n_heads=n_heads,
                             d_ff=d_ff,
                             moving_avg=moving_avg,
                             dropout_rate=dropout_rate,
                             activation=activation)
                for _ in range(n_layers)
        ]
        self.norm_layer = layers.Normalization([-2,-1])
        self.out_proj = layers.Dense(d_out, activation="linear",
                                     use_bias=True)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dec_layers": self.dec_layers,
                "norm_layer": self.norm_layer
            }
        )

    def call(self, inputs):
        x, cross, xt = inputs
        for dec_layer in self.dec_layers:
            x, residual_trend = dec_layer([x,cross])
            print(residual_trend.shape.as_list())
            xt += residual_trend

        x = self.norm_layer(x)
        x = self.out_proj(x)

        return x, xt

@keras.saving.register_keras_serializable(package="Autoformer")
class Autoformer(keras.Model):
    """
    Keras implementation of the Autoformer
    """
    def __init__(self, config:dict, **kwargs):
        super(Autoformer, self).__init__(**kwargs)

        # Model hyper-parameters
        self.d_model = config["d_model"] # embedding depth
        self.dropout_rate = config["dropout_rate"] # self-explanatory
        self.d_ff = config["d_ff"] # dimensionality of the feed-forward layers
        self.d_out = config["d_out"] # number of output channels

        # Encoder/Decoder dimensionality
        self.n = config["N"] # number of Encoder Layers
        self.m = config["M"] # number of Decoder Layers

        # Multi-head dimensionality
        self.k = config["k"] # top_k = k * log(L)
        self.h = config["h"] # number of heads

        # Time parameters
        self.o = config["O"] # prediction horizon
        self.tau = config["tau"] # moving average window


        self.embed = DataEmbedding(d_model=self.d_model,
                                   dropout_rate=self.dropout_rate)

        self.moving_avg = layers.AvgPool1D(pool_size=self.tau,
                                           strides=1,
                                           padding="same")

        self.encoder = CorrEncoder(n_layers=self.n,
                                   k_factor=self.k,
                                   n_heads=self.h,
                                   d_ff=self.d_ff,
                                   moving_avg=self.tau,
                                   dropout_rate=self.dropout_rate)

        self.decoder = CorrDecoder(d_out=self.d_out,
                                   n_layers=self.m,
                                   k_factor=self.k,
                                   n_heads=self.h,
                                   d_ff=self.d_ff,
                                   moving_avg=self.tau,
                                   dropout_rate=self.dropout_rate)

    def _init_decoder_input(self, x_enc:tf.Tensor,
                            x_enc_marks:tf.Tensor,
                            x_dec_marks:tf.Tensor):
        _, l, _ = x_enc.shape
        half_input = x_enc[:,l//2:,:]
        half_trend = self.moving_avg(half_input)
        half_season = half_input - half_trend

        x_dec_s = tf.pad(half_season, tf.constant([[0,0],[0,self.o],[0,0]]),
                         mode="CONSTANT", constant_values=0)
        x_dec_t = tf.concat([half_trend,
                        tf.tile(
                            tf.math.reduce_mean(x_enc, axis=1, keepdims=True),
                            [1,self.o,1])
                        ],
                        axis=1)
        half_marks = x_enc_marks[:,l//2:,:]
        x_dec_marks = tf.concat([half_marks, x_dec_marks],
                                    axis=1)

        return x_dec_s, x_dec_t, x_dec_marks

    def call(self, inputs):
        # Enc Input -> Dec Input
        x_enc, x_enc_marks, x_dec_marks = inputs
        x_dec_s, x_dec_t, x_dec_marks = \
            self._init_decoder_input(x_enc, x_enc_marks, x_dec_marks)

        # Apply embedding
        x_enc = self.embed([x_enc, x_enc_marks])
        x_dec_s = self.embed([x_dec_s, x_dec_marks])

        # Enc call
        y_enc = self.encoder(x_enc)
        # Dec call
        y_dec_s, y_dec_t = self.decoder([x_dec_s,y_enc,x_dec_t])

        # Trend+Seasonality join
        y_dec = y_dec_s + y_dec_t

        return y_dec
