# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/


from keras import layers, Model, activations


def conv_block(x, n_filters):
    x = layers.Conv2D(filters=n_filters, kernel_size=3, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def downsample_block(x, n_filters, dropout=None):
    f = conv_block(x, n_filters)
    p = layers.MaxPool2D(pool_size=2)(f)
    if dropout is not None:
        p = layers.Dropout(dropout)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout=None):
    x = layers.Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = conv_block(x, n_filters)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    return x


def calculate_kernel_size(input_shape, target_shape):
    kernel_size = []
    for i in range(len(input_shape)):
        kernel_size.append(input_shape[i] - target_shape[i] + 1)
    return tuple(kernel_size)


def final_block(x, final_channel, out_channels, digits, dropout=None):
    kernel_size = calculate_kernel_size((16, 16), (digits, 10))
    x = layers.Conv2D(final_channel, kernel_size=kernel_size)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(out_channels, kernel_size=3, padding="same")(x)
    x = activations.sigmoid(x)
    return x


def model(image_shape, in_channels, out_channels, digits=10, channels=[16, 32, 64, 128], final_channel=32,  dropout=None):
    inputs = layers.Input(shape=image_shape + (in_channels,))

    f1, p1 = downsample_block(inputs, channels[0], dropout=dropout)  # 128
    f2, p2 = downsample_block(p1, channels[0], dropout=dropout)  # 64
    f3, p3 = downsample_block(p2, channels[1], dropout=dropout)  # 32
    f4, p4 = downsample_block(p3, channels[1], dropout=dropout)  # 16
    f5, p5 = downsample_block(p4, channels[2], dropout=dropout)  # 8
    f6, p6 = downsample_block(p5, channels[2], dropout=dropout)  # 4
    f7, p7 = downsample_block(p6, channels[2], dropout=dropout)  # 2

    bottleneck = conv_block(p7, channels[3])  # 1

    u6 = upsample_block(bottleneck, f7, channels[2], dropout=dropout)  # 2
    u7 = upsample_block(u6, f6, channels[2], dropout=dropout)  # 4
    u8 = upsample_block(u7, f5, channels[2], dropout=dropout)  # 8
    u9 = upsample_block(u8, f4, channels[1], dropout=dropout)  # 16

    outputs = final_block(u9, final_channel, out_channels, digits, dropout)

    model = Model(inputs, outputs, name="Modified U-Net")
    return model
