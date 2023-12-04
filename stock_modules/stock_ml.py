"""
Python module handling model creation and training.
"""

import tensorflow as tf

def create_direction_prediction_model(m, n):
    """ Create a model which predicts the direction of the stock price.
    The direction is up, down or flat for each stock,
    so the output is a (nx3) matrix.
    """
    inputs = tf.keras.layers.Input(shape=(m, n))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(16, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # The output is a (nx3) matrix, where column is a one-hot vector (probability distribution over the classes)
    x = tf.keras.layers.Dense(n*3, activation="relu")(x)
    outputs = tf.keras.layers.Reshape((3,n))(x)
    # Apply softmax to each column
    outputs = tf.keras.layers.Softmax(axis=1)(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_updown_prediction_model_dense(m, n, output_scale=(0, 1)):
    """ Dense model"""
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = tf.keras.layers.Input(shape=(m, n))
    
    # Normalize input data
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(n, activation="tanh" if output_scale == (-1, 1) else "sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model
    

def create_updown_prediction_model(m, n, output_scale=(0, 1)):
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = tf.keras.layers.Input(shape=(m, n))
    
    # Normalize input data
    x = tf.keras.layers.BatchNormalization()(inputs)

    # Bidirectional LSTM layers with dropout
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Attention layer
    x = tf.keras.layers.Attention()([x, x, x])

    # Flatten and add dropout
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Dense layers
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n, activation="tanh" if output_scale == (-1, 1) else "sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_updown_prediction_model_rnn(m, n, output_scale=(0, 1)):
    if output_scale not in [(0, 1), (-1, 1)]:
        raise ValueError("output_scale must be either (0,1) or (-1,1).")

    inputs = tf.keras.layers.Input(shape=(m, n))
    
    # Normalize input data
    x = tf.keras.layers.BatchNormalization()(inputs)

    # Bidirectional LSTM layers with dropout
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.SimpleRNN(64)(x)

    x = tf.keras.layers.Dropout(0.2)(x)

    # Attention layer
    x = tf.keras.layers.Attention()([x, x, x])

    # Flatten and add dropout
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Dense layers
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    # Output layer
    outputs = tf.keras.layers.Dense(n, activation="tanh" if output_scale == (-1, 1) else "sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(f"Model summary: {model.summary()}")

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_price_prediction_model(m, n):
    """
    Create an LSTM model which takes in m values of n stocks (m x n), and
    outputs the predicted value for each stock in the next time step (1 x n).
    m is the batch size, and n is the number of stocks.
    """
    inputs = tf.keras.layers.Input(shape=(m, n))
    #x = tf.keras.layers.BatchNormalization()(inputs)
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
