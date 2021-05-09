
import tensorflow as tf

import tensorflow.keras.models as models
from encoder import Encoder
from decoder import Decoder
from utils import reparameterize
from losses import kl

class NVAE(models.Model):
    def __init__(self, z_dim):
        super(NVAE, self).__init__()

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def call(self, x):

        mu, log_var, xs = self.encoder(x)

        z = reparameterize(mu, tf.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        # TODO
        #recon_loss = tf.reduce_mean(decoder_output - x)

        kl_loss = kl(mu, log_var)

        return decoder_output, [kl_loss] + losses