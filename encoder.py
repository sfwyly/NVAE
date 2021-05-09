
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

from common import EncoderResidualBlock, Swish


class ConvBlock(layers.Layer):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self._seq = models.Sequential([
            layers.Conv2D(out_channels, kernel_size=3, padding="same"), # in_channels -> out_channels
            layers.Conv2D(out_channels // 2, kernel_size  =1),
            layers.BatchNormalization(), Swish(),
            layers.Conv2D(out_channels, kernel_size=3, strides  = 2, padding = "same"),
            layers.BatchNormalization(), Swish()
        ])

    def call(self, x):

        return self._seq(x)


class EncoderBlock(layers.Layer):

    def __init__(self, channels):

        super(EncoderBlock, self).__init__()

        self.channels = channels
        self.modules = []

        for i in range(len(channels) - 1):
            self.modules.append(ConvBlock(channels[i], channels[i+1]))
    def call(self, x):

        for module in self.modules:
            x = module(x)
        return x

class Encoder(layers.Layer):

    def __init__(self, z_dim):

        super(Encoder, self).__init__()
        self.encoder_blocks = [
            EncoderBlock([3, z_dim //16, z_dim //8]),
            EncoderBlock([z_dim//8, z_dim//4, z_dim//2]),
            EncoderBlock([z_dim//2, z_dim])
        ]

        self.encoder_residual_blocks = [
            EncoderResidualBlock(z_dim // 8),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim)
        ]

        self.condition_x = models.Sequential([
            Swish(), layers.Conv2D(z_dim *2, kernel_size = 1) # z_dim -> z_dim * 2
        ])

    def call(self, x):
        xs = []
        last_x = x

        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)

        mu, log_var = tf.split(self.condition_x(last_x), 2, -1)

        return mu, log_var, xs[:-1][::-1]

