import rn10
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import os

BATCH_SIZE = 128
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
NUM_CLASSES = 10
CHANNELS = 3

class BasicBlock(tf.keras.layers.Layer):  
  def __init__(self, filter_num, stride=1):
    super(BasicBlock, self).__init__()
    initializer = tf.keras.initializers.HeNormal()
    l2 = tf.keras.regularizers.l2(0.0001)
    self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3), 
                  strides=stride, padding='same', kernel_regularizer=l2, 
                  kernel_initializer=initializer)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3,3),
                  strides=1, padding='same', kernel_regularizer=l2, 
                  kernel_initializer=initializer)
    self.bn2 = tf.keras.layers.BatchNormalization()
    if stride != 1:
      self.downsample = tf.keras.Sequential()
      self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1,1),
                  strides=stride, kernel_regularizer=l2,
                  kernel_initializer=initializer))
      self.downsample.add(tf.keras.layers.BatchNormalization())
    else:
      self.downsample = lambda x: x

  def call(self, inputs, **kwargs):
    residual = self.downsample(inputs)

    x = self.conv1(inputs)
    x = self.bn1(x)
    x = tf.nn.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)

    output = tf.nn.relu(tf.keras.layers.add([residual, x]))

    return output

def make_basic_block_layer(filter_num, blocks, stride=1):
  res_block = tf.keras.Sequential()
  res_block.add(BasicBlock(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(BasicBlock(filter_num, stride=1))

  return res_block

class ResNet(tf.keras.Model):  
  def __init__(self, layer_params):
    super(ResNet, self).__init__()
    initializer = tf.keras.initializers.HeNormal()
    l2 = tf.keras.regularizers.l2(0.0001)
    self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, 
                    padding='same', kernel_regularizer=l2, kernel_initializer=initializer)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.layer1 = make_basic_block_layer(filter_num=16, blocks=layer_params[0], stride=2)
    self.layer2 = make_basic_block_layer(filter_num=32, blocks=layer_params[1], stride=2)
    self.layer3 = make_basic_block_layer(filter_num=64, blocks=layer_params[2], stride=2)
    self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
    self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax,
                    kernel_regularizer=l2, kernel_initializer=initializer)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = tf.nn.relu(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.avgpool(x)
    output = self.fc(x)

    return output

def get_training_model1():
  model = ResNet(layer_params=[3, 3, 3]) # basic block으로 3쌍의 conv-bn layer를 쌓음
  model.build(input_shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=False), 
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

def get_training_model2():
    n = 2
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
