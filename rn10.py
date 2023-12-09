import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.layers import AveragePooling2D, Flatten

def stem(inputs):
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
    
def learner(x, n_blocks):
    x = residual_group(x, 16, n_blocks, strides=(1, 1), n=4)

    x = residual_group(x, 64, n_blocks, n=2)

    x = residual_group(x, 64, n_blocks, n=2)

    x = residual_group(x, 64, n_blocks, n=2)

    x = residual_group(x, 128, n_blocks, n=2)

    x = residual_group(x, 128, n_blocks, n=2)

    x = residual_group(x, 128, n_blocks, n=2)
    return x

def residual_group(x, n_filters, n_blocks, strides=(2, 2), n=2):
    x = projection_block(x, n_filters, strides=strides, n=n)

    for _ in range(n_blocks):
        x = identity_block(x, n_filters, n)
    return x
    
def identity_block(x, n_filters, n=2):
    shortcut = x

    x = Conv2D(n_filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def projection_block(x, n_filters, strides=(2,2), n=2):
    shortcut = Conv2D(n_filters * n, (1, 1), strides=strides, kernel_initializer='he_normal')(x)

    x = Conv2D(n_filters, (1, 1), strides=(1,1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)  
    
    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
    
def classifier(x, n_classes):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=8)(x)
    
    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

n = 18
depth =  n * 9 + 2
n_blocks = ((depth - 2) // 9) - 1

inputs = Input(shape=(32, 32, 3))

x = stem(inputs)
   
x = learner(x, n_blocks)

outputs = classifier(x, 10)

model = Model(inputs, outputs)
