"""
Python module handling model creation and training.
"""

import tensorflow as tf

def create_tf_model(m, n):
  """
  Create an LSTM model which takes in m values of n stocks (m x n), and
  outputs the predicted value for each stock in the next time step (1 x n).
  m is the batch size, and n is the number of stocks.
  """
  inputs = tf.keras.layers.Input(shape=(m, n))
  x = tf.keras.layers.BatchNormalization()(inputs)
  # Bidirectional LSTM layers
  x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer="l2")
      )(inputs)
  x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(16, return_sequences=True)
      )(x)

  # Attention layer
  x = tf.keras.layers.Attention()([x, x, x])

  # Flatten and add dropout
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dropout(0.5)(x)

  # Dense layer
  x = tf.keras.layers.Dense(32, activation="relu")(x)

  # Output (1,N)
  outputs = tf.keras.layers.Dense(n, activation="linear")(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  print(f"Model summary: {model.summary()}")
  # Compile the model
  model.compile(optimizer="adam", loss="huber")
  return model
