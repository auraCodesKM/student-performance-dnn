"""Step 2: the deep neural network (DNN), built with Keras."""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras runs on top of PyTorch here (no TensorFlow needed)

import keras
from keras import layers


def build_model(n_inputs, hidden=(16, 8), dropout=0.3, learning_rate=0.001, activation="relu", optimizer="adam"):
    """inputs -> hidden layers (each: Dense + Dropout) -> 1 output with sigmoid = probability of being at risk."""
    keras.utils.set_random_seed(42)  # same starting weights every time, so results are repeatable
    model = keras.Sequential([keras.Input(shape=(n_inputs,))])
    for units in hidden:
        model.add(layers.Dense(units, activation=activation, kernel_initializer="he_normal"))
        model.add(layers.Dropout(dropout))  # randomly switches off neurons while training, to avoid overfitting
    model.add(layers.Dense(1, activation="sigmoid"))  # output between 0 and 1
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X, y):
    """Train with early stopping: stop when the validation loss hasn't improved for 20 epochs, keep the best weights."""
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    return model.fit(X, y, validation_split=0.15, epochs=300, batch_size=32, callbacks=[stop], verbose=0)


def predict_at_risk(model, X):
    """Probability of being at risk for each student."""
    return model.predict(X, verbose=0)[:, 0]
