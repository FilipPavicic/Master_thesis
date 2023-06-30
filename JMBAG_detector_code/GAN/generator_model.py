# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Lambda


def conv_block(x, n_filters, kernel_initializer='glorot_uniform'):
    # Conv2D then ReLU activation
    x = keras.layers.Conv2D(n_filters, 3, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(x)
    return x


def downsample_block(x, n_filters, dropout=None, kernel_initializer='glorot_uniform'):
    f = conv_block(x, n_filters, kernel_initializer)
    p = keras.layers.MaxPool2D(2)(f)
    if dropout is not None:
        p = keras.layers.Dropout(dropout)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout=None):
    # upsample
    x = keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = keras.layers.concatenate([x, conv_features])
    # dropout
    if dropout is not None:
        x = keras.layers.Dropout(dropout)(x)
    # Conv2D twice with ReLU activation
    x = conv_block(x, n_filters)
    return x


def model(image_shape,  in_channels, out_channels, label_shape=(10, 10), channels=[8, 8, 16, 16, 32], dropout=None, kernel_initializer='glorot_uniform'):
    # inputs
    input = keras.layers.Input(shape=image_shape + (in_channels,))
    label = keras.layers.Input(shape=label_shape + (1,))
    random_layer = keras.layers.Input(shape=label_shape + (1,))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(input, channels[0], dropout=dropout, kernel_initializer=kernel_initializer)
    # 2 - downsample
    f2, p2 = downsample_block(p1, channels[1], dropout=dropout, kernel_initializer=kernel_initializer)
    # 3 - downsample
    f3, p3 = downsample_block(p2, channels[2], dropout=dropout, kernel_initializer=kernel_initializer)
    # 4 - downsample
    bottleneck = keras.layers.Conv2D(channels[3], 7, activation=keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(p3)
    expanded_label = Lambda(lambda x: tf.tile(x, [1, 1, 1, channels[3]]))(label)
    concatenated_bottlecenk = keras.layers.concatenate([bottleneck, expanded_label])
    u4 = keras.layers.Conv2DTranspose(channels[3], 7)(concatenated_bottlecenk)

    # 6 - upsample
    u5 = upsample_block(u4, f3, channels[2], dropout=dropout)
    # 7 - upsample
    u6 = upsample_block(u5, f2, channels[1], dropout=dropout)
    # 8 - upsample
    u7 = upsample_block(u6, f1, channels[0], dropout=dropout)
    # 9 - upsample
    output = keras.layers.Conv2D(out_channels, 3, padding="same", activation="sigmoid")(u7)

    # unet model with Keras Functional API
    model = keras.Model([input, label, random_layer], output, name="Generator")
    return model
