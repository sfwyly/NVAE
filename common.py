

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class GroupConv2D(layers.Layer):

    def __init__(self, filters, kernel_size = 3, strides = 1, dilation_rate = 1, padding = "same", groups = 1):

        super(GroupConv2D, self).__init__()

        assert filters % groups ==0 , 'filters 不能被 groups整除'

        self.groups = groups

        self.models = []
        per_filters = filters // groups

        for _ in range(groups):
            self.models.append(layers.Conv2D(per_filters, kernel_size = kernel_size, strides=strides, dilation_rate=dilation_rate,padding=padding))

    def call(self, x):

        x_children = tf.split(x, self.groups, -1)

        r = []
        for i,x_child in enumerate(x_children):
            r.append(self.models[i](x_child))
        return tf.concat(r, axis = -1)





class SELayer(layers.Layer):

    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = models.Sequential([
            layers.Dense(channel // reduction, use_bias = False),
            layers.ReLU(),
            layers.Dense(channel, use_bias=False),
            layers.Activation("sigmoid")
        ])

    def call(self, x):
        b, _, _, c = x.shape
        y = self.avg_pool(x)
        y = self.fc(y)
        y = tf.reshape(y,(b,1,1,c))
        return x * y

class Swish(layers.Layer):

    def __init__(self):
        super(Swish, self).__init__()
    def call(self, x):
        return x * tf.nn.sigmoid(x)

class ResidualBlock(layers.Layer):

    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self._seq = models.Sequential([
            layers.Conv2D(dim, kernel_size = 5, padding="same"), # dim -> dim
            layers.Conv2D(dim, kernel_size = 1), # dim-> dim
            layers.BatchNormalization(),Swish(),
            layers.Conv2D(dim, kernel_size = 3, padding="same"),
            SELayer(dim)
        ])
    def call(self, x):
        return x + 0.1 * self._seq(x)

class EncoderResidualBlock(layers.Layer):

    def __init__(self, dim):
        super(EncoderResidualBlock, self).__init__()
        self.seq = models.Sequential([
            layers.Conv2D(dim, kernel_size = 5, padding="same"), # dim -> dim
            layers.Conv2D(dim, kernel_size = 1), # dim-> dim
            layers.BatchNormalization(),Swish(),
            layers.Conv2D(dim, kernel_size = 3, padding="same"), # dim -> dim
            SELayer(dim)
        ])
    def call(self, x):
        return x + 0.1 * self.seq(x)

class DecoderResidualBlock(layers.Layer):

    def __init__(self, dim, n_groups):
        super(DecoderResidualBlock, self).__init__()

        self._seq = models.Sequential([
            layers.Conv2D(n_groups * dim, kernel_size = 1), # dim -> n_groups_dim
            layers.BatchNormalization(), Swish(),
            #layers.Conv2D(n_group * dim, kernel_size = 5, padding="same",n_group=n_group),
            GroupConv2D(n_groups * dim, kernel_size=5, padding="same",groups=n_groups),
            layers.BatchNormalization(), Swish(),
            layers.Conv2D(dim, kernel_size=1),
            layers.BatchNormalization(),
            SELayer(dim)
        ])

    def call(self, x):

        return x + 0.1 * self._seq(x)

