
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from common import Swish, DecoderResidualBlock, ResidualBlock
from utils import reparameterize
from losses import kl_2

class UpsampleBlock(layers.Layer):

    def __init__(self, in_channels, out_channel):

        super(UpsampleBlock, self).__init__()

        self._seq = models.Sequential([
            layers.Conv2DTranspose(out_channel, kernel_size=3, strides = 2,padding="same"), # in_channeks -> out_channels
            layers.BatchNormalization(),
            Swish()
        ])

    def call(self, x):
        return self._seq(x)

class DecoderBlock(layers.Layer):

    def __init__(self, channels):
        super(DecoderBlock, self).__init__()

        self.channels = channels
        self.modules = []

        for i in range(len(channels) - 1):
            self.modules.append(UpsampleBlock(channels[i], channels[i+1]))

    def call(self, x):

        for module in self.modules:
            x = module(x)
        return x

class Decoder(layers.Layer):

    def __init__(self, z_dim):

        super(Decoder, self).__init__()

        self.decoder_blocks = [
            DecoderBlock([z_dim * 2, z_dim //2]),
            DecoderBlock([z_dim, z_dim // 4, z_dim // 8]),
            DecoderBlock([z_dim //4, z_dim // 16, z_dim // 32])
        ]

        self.decoder_residual_blocks = [
            DecoderResidualBlock(z_dim // 2, n_groups = 4),
            DecoderResidualBlock(z_dim // 8, n_groups = 2),
            DecoderResidualBlock(z_dim // 32, n_groups = 1)
        ]

        self.condition_z = [
            models.Sequential([
                ResidualBlock(z_dim // 2),
                Swish(),
                layers.Conv2D(z_dim,kernel_size=1) # z_dim // 2 -> z_dim
            ]),
            models.Sequential([
                ResidualBlock(z_dim // 8),
                Swish(),
                layers.Conv2D(z_dim // 4, kernel_size=1)  # z_dim // 8 -> z_dim//4
            ]),
        ]

        # p(z_1 | x, z_(l - 1))
        self.condition_xz = [
            models.Sequential([
                ResidualBlock(z_dim),
                layers.Conv2D(z_dim //2, kernel_size = 1),# z_dim -> z_dim//2
                Swish(),
                layers.Conv2D(z_dim, kernel_size = 1) # z_dim // 2 -> z_dim
            ]),
            models.Sequential([
                ResidualBlock(z_dim //4),
                layers.Conv2D(z_dim // 8, kernel_size=1), # z_dim // 4 -> z_dim // 8
                Swish(),
                layers.Conv2D(z_dim // 4, kernel_size = 1) # z_dim // 8 -> z_dim // 4
            ])
        ]

        self.recon = models.Sequential([
            ResidualBlock(z_dim // 32),
            layers.Conv2D(3, kernel_size = 1)# z_dim // 32 -> 3
        ])

    def call(self, z, xs=None):
        """
        :param z: shape = (B, z_dim, map_h, map_w)
        :param xs:
        :return:
        """
        B, map_h, map_w, D = z.shape

        decoder_out = tf.zeros((B,map_h,map_w,D), dtype=z.dtype)

        kl_losses = []

        for i in range(len(self.decoder_residual_blocks)):

            z_sample = tf.concat([decoder_out, z], axis = -1)

            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if(i == len(self.decoder_residual_blocks) - 1):
                break

            mu, log_var = tf.split(self.condition_z[i](decoder_out), 2, -1)

            if(xs is not None):

                delta_mu, delta_log_var = tf.split(self.condition_xz[i](tf.concat([decoder_out, xs[i]], axis = -1)), num_or_size_splits = 2, axis = -1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            z = reparameterize(mu, tf.exp(0.5 * log_var))
            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        x_hat = tf.nn.sigmoid(self.recon(decoder_out))

        return x_hat, kl_losses





