from tensorflow import keras


def conv_block(x, n_filters, kernel_initializer='glorot_uniform'):
    # Conv2D then ReLU activation
    x = keras.layers.Conv2D(n_filters, 3, 2, padding="same", activation=keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=kernel_initializer)(x)
    return x


def model(image_shape, in_channels, label_shape=(10, 10), up_channels=[16, 32, 2], channels=[16, 16, 32, 32, 64, 64, 64], dropout=None, kernel_initializer='glorot_uniform'):

    label_input = keras.layers.Input(shape=label_shape + (1,))
    image_input = keras.layers.Input(shape=image_shape + (in_channels,))
    u16 = keras.layers.Conv2DTranspose(up_channels[0], 7)(label_input)
    # upsample
    u64 = keras.layers.Conv2DTranspose(up_channels[1], 3, 4, padding="same")(u16)
    # upsample
    u128 = keras.layers.Conv2DTranspose(up_channels[2], 3, 2, padding="same")(u64)

    concatenated_layer128 = keras.layers.concatenate([u128, image_input])

    fe64 = conv_block(concatenated_layer128, channels[0])
    fe32 = conv_block(fe64, channels[1])
    fe16 = conv_block(fe32, channels[2])
    fe8 = conv_block(fe16, channels[3])
    fe4 = conv_block(fe8, channels[4])
    fe2 = conv_block(fe4, channels[5])
    fe1 = conv_block(fe2, channels[6])
    dense = keras.layers.Flatten()(fe1)
    if dropout is not None:
        dense = keras.layers.Dropout(dropout)(dense)
    output = keras.layers.Dense(1)(dense)

    return keras.Model([image_input, label_input], output, name="discriminator")
