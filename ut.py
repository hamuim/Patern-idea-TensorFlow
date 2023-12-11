import rn10
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

def get_training_model():
    n = 12
    depth =  n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    x = rn10.stem(inputs)

    x = rn10.learner(x, n_blocks)

    outputs = rn10.classifier(x, 10)

    model = tf.keras.Model(inputs, outputs)
    
    return model

def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
