from tensorflow import keras
from keras import layers


def conv_block(x, n_filters, kernel_initializer='glorot_uniform'):
    # Conv2D then ReLU activation
    x  # = layers.Conv2D(n_filters, 3, 1, padding="same", activation=layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(n_filters, 3, 2, padding="same", activation=layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    return x


def model(image_shape, in_channels, label_shape=(10, 10), up_channels=[16, 32, 2], channels=[16, 16, 32, 32, 64, 64, 64], dropout=None, kernel_initializer='glorot_uniform'):

    label_input = layers.Input(shape=label_shape + (1,))
    image_input = layers.Input(shape=image_shape + (in_channels,))
    u16 = layers.Conv2DTranspose(up_channels[0], 7)(label_input)
    # upsample
    u64 = layers.Conv2DTranspose(up_channels[1], 3, 4, padding="same")(u16)
    # upsample
    u128 = layers.Conv2DTranspose(up_channels[2], 3, 2, padding="same")(u64)

    concatenated_layer128 = layers.concatenate([u128, image_input])

    fe64 = conv_block(concatenated_layer128, channels[0])
    fe32 = conv_block(fe64, channels[1])
    fe16 = conv_block(fe32, channels[2])
    output = layers.Conv2D(1, 3, 2, padding="same", activation="sigmoid", kernel_initializer=kernel_initializer)(fe16)

    return keras.Model([image_input, label_input], output, name="discriminator")
