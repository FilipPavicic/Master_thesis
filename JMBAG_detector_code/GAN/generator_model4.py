# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

from tensorflow import keras
from keras import layers


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


def model(image_shape,  in_channels, out_channels, label_shape=(10, 10), concat_channels=[32, 32, 32, 3], channels=[8, 8, 16, 16, 32], dropout=None, kernel_initializer='glorot_uniform'):
    # inputs
    backround = layers.Input(shape=image_shape + (in_channels,))
    label = layers.Input(shape=label_shape + (1,))
    random_layer = layers.Input(shape=label_shape + (1,))

    concat10 = layers.concatenate([label, random_layer])
    concat16 = layers.Conv2DTranspose(concat_channels[0], kernel_size=7,  activation=keras.layers.LeakyReLU(alpha=0.2))(concat10)
    concat32 = layers.Conv2DTranspose(concat_channels[1], kernel_size=4, strides=2, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2))(concat16)
    concat64 = layers.Conv2DTranspose(concat_channels[2], kernel_size=4, strides=2, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2))(concat32)
    concat128 = layers.Conv2DTranspose(concat_channels[3], kernel_size=4, strides=2, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2))(concat64)

    input = layers.concatenate([backround, concat128])
    x = layers.Conv2D(channels[0], 4, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2))(input)
    output = layers.Conv2D(out_channels, 4, padding="same", activation="sigmoid")(x)

    # unet model with Keras Functional API
    model = keras.Model([backround, label, random_layer], output, name="Generator")
    return model
