# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

from tensorflow import keras
from keras import layers, Model


def conv_block(x, n_filters, batch_normalization=False):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, kernel_size=3, padding="same")(x)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    #x = layers.Conv2D(n_filters, 5, padding="same")(x)
    # if batch_normalization:
    #     x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def downsample_block(x, n_filters, dropout=None, batch_normalization=False):
    f = conv_block(x, n_filters, batch_normalization=batch_normalization)
    p = layers.MaxPool2D(2)(f)
    if dropout is not None:
        p = layers.Dropout(dropout)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout=None, batch_normalization=False):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # Conv2D twice with ReLU activation
    x = conv_block(x, n_filters, batch_normalization=batch_normalization)
    return x


def model(image_shape, in_channels, out_channels, channels=[16, 32, 64, 128], final_channel=32,  dropout=None, kernel_initializer='glorot_uniform', batch_normalization=False):
    # inputs
    inputs = layers.Input(shape=image_shape + (in_channels,))
    # encoder: contracting path - downsample
    # 0 - downsample
    f1, p1 = downsample_block(inputs, channels[0], dropout=dropout, batch_normalization=batch_normalization)  # 128
    # 2 - downsample
    f2, p2 = downsample_block(p1, channels[0], dropout=dropout, batch_normalization=batch_normalization)  # 64
    # 3 - downsample
    f3, p3 = downsample_block(p2, channels[1], dropout=dropout, batch_normalization=batch_normalization)  # 32
    # 4 - downsample
    f4, p4 = downsample_block(p3, channels[1], dropout=dropout, batch_normalization=batch_normalization)  # 16
    # 5 - downsample
    f5, p5 = downsample_block(p4, channels[2], dropout=dropout, batch_normalization=batch_normalization)  # 8
    # 6 - downsample
    f6, p6 = downsample_block(p5, channels[2], dropout=dropout, batch_normalization=batch_normalization)  # 4
    # 7 - downsample
    f7, p7 = downsample_block(p6, channels[2], dropout=dropout, batch_normalization=batch_normalization)  # 2
    # 7 - bottleneck
    bottleneck = conv_block(p7, channels[3], batch_normalization=batch_normalization)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f7, channels[2], dropout=dropout, batch_normalization=batch_normalization)
    # 7 - upsample
    u7 = upsample_block(u6, f6, channels[2], dropout=dropout, batch_normalization=batch_normalization)
    # 8 - upsample
    u8 = upsample_block(u7, f5, channels[2], dropout=dropout, batch_normalization=batch_normalization)
    # 9 - upsample
    u9 = upsample_block(u8, f4, channels[1], dropout=dropout, batch_normalization=batch_normalization)
    # outputs
    u10 = layers.Conv2D(final_channel, kernel_size=7)(u9)
    u10 = layers.LeakyReLU(alpha=0.2)(u10)
    if batch_normalization:
        u10 = layers.BatchNormalization()(u10)
    if dropout is not None:
        u10 = layers.Dropout(dropout)(u10)
    outputs = layers.Conv2D(out_channels, kernel_size=3, padding="same", activation=keras.activations.sigmoid)(u10)
    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model
