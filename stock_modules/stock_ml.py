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

class SkipTDLoss(tf.keras.losses.Loss):
    """ A loss funciton, that calculates the given loss but ignores the first column from y_true and y_pred.
    """
    def __init__(self, base_loss_fun, **kwargs):
        super().__init__(**kwargs)
        self.base_loss_fun = base_loss_fun
    
    def call(self, y_true, y_pred):
        # Skip first column
        return self.base_loss_fun(y_true[:,1:], y_pred[:,1:])
    
class MultiSoftmaxLoss(tf.keras.losses.Loss):
    """ Calculates the mean categorical crossentropy loss for each stock prediction.
    So input is (batch_size, nhours, nstocks), and output is (batch_size, 3, nstocks)
    The CCE/log loss is calculated for each stock and then averaged.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fun = tf.keras.losses.CategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        pred_losses = []
        # Calculate the softmax loss for each one-hot prediction
        # Y data is (3, nstocks), so take one column at a time, compare it to the true value and calculate the loss
        for i in range(y_true.shape[2]):
            y_pred_i = y_pred[:,:,i]
            y_true_i = y_true[:,:,i]
            #print(f"Comparing y_true_i shape: {y_true_i.shape} to y_pred_i shape: {y_pred_i.shape}")
            pred_losses.append(self.loss_fun(y_true_i, y_pred_i))
        # Now we have a list of losses for each stock, so we can calculate the mean
        pred_losses = tf.stack(pred_losses, axis=0)
        mean_loss = tf.reduce_mean(pred_losses, axis=0)
        return mean_loss
    
class MultiAccuracy(tf.keras.metrics.Metric):
    """ Calculates the average prediction accuracy for the predictions.
    So for stock, we calculate the accuracy of the prediction, sum them and then divide by the number of stocks.
    """
    def __init__(self, has_timedelta = False, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = 0 if not has_timedelta else 1
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate the accuracy for each stock
        accs = []
        for i in range(self.start_idx, y_true.shape[2]):
            y_true_i = y_true[:,:,i]
            y_pred_i = y_pred[:,:,i]
            # y_pred_i to onehot
            y_pred_i = tf.one_hot(tf.argmax(y_pred_i, axis=1), depth=3)
            accs.append(self.accuracy(y_true_i, y_pred_i))
        # Now we have a list of accuracies for each stock, so we can calculate the mean
        accs = tf.stack(accs, axis=0)
        mean_acc = tf.reduce_mean(accs, axis=0)
        self.mean_acc = mean_acc
    
    def result(self):
        return self.mean_acc
    
    def reset_states(self):
        pass
