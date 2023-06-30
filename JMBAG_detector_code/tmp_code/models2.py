# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

from tensorflow import keras
from keras import layers, Model


def conv_block(x, n_filters, kernel_initializer='glorot_uniform'):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation=layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(x)
    return x


def downsample_block(x, n_filters, dropout=None, kernel_initializer='glorot_uniform'):
    f = conv_block(x, n_filters, kernel_initializer)
    p = layers.MaxPool2D(2)(f)
    if dropout is not None:
        p = layers.Dropout(dropout)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout=None):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # Conv2D twice with ReLU activation
    x = conv_block(x, n_filters)
    return x


def model(image_shape, in_channels, out_channels, channels=[64, 128, 256], dropout=None, kernel_initializer='glorot_uniform'):
    # inputs
    inputs = layers.Input(shape=image_shape + (in_channels,))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, channels[0], dropout=dropout, kernel_initializer=kernel_initializer)
    # 2 - downsample
    f2, p2 = downsample_block(p1, channels[1], dropout=dropout, kernel_initializer=kernel_initializer)
    # 3 - downsample
    f3, p3 = downsample_block(p2, channels[2], dropout=dropout, kernel_initializer=kernel_initializer)
    # 4 - downsample
    outputs = layers.Conv2D(out_channels, 7, activation=keras.activations.sigmoid)(p3)
    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model
