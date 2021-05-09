
from model import NVAE
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class Trainer():

    def __init__(self, z_dim = 512):
        # nvae = NVAE(z_dim=512)
        #
        # inputs = layers.Input(batch_shape=(6,256,256,3))
        # outputs = nvae(inputs)
        # model = models.Model(inputs, outputs)
        # model.summary()
        self.z_dim = z_dim
        self.build_model()

        # style loss
        style_layers = ['block1_pool', 'block2_pool', 'block3_pool']
        self.vgg = self.vgg_layers(style_layers)

    def save_weights(self):
        self.nvae.save_weights("/root/sfwy/inpainting/scf/nvae.h5")

    def load_weights(self):
        self.nvae.load_weights("E:/future/scf/nvae.h5")
        # self.nvae.load_weights("/root/sfwy/inpainting/scf/nvae.h5")

    def build_model(self):
        self.nvae = NVAE(z_dim = self.z_dim)

        self.nvae.build(input_shape=(6,64,64,3))
        # nvae = NVAE(z_dim=self.z_dim)
        #
        # inputs = layers.Input(batch_shape=(6,256,256,3))
        # outputs = nvae(inputs)
        # self.nvae = models.Model(inputs, outputs)
        # self.nvae.summary()

    def compute_vgg_loss(self, vgg, img, target):

        img_vgg = img #vgg_preprocess(img)
        target_vgg = target #vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        loss = 0.
        for img_f, target_f in zip(img_fea, target_fea):
            loss += tf.reduce_mean(tf.abs(img_f - target_f))
        return loss

    # calc style loss
    def compute_style_loss(self, vgg, img, target):
        img_vgg = img #vgg_preprocess(img)
        target_vgg = target #vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        loss = 0.
        for img_f, target_f in zip(img_fea, target_fea):
            b, h, w, c = img_f.shape
            img_n = tf.linalg.einsum("bijc,bijd->bcd", img_f, img_f)/(h*w*c)
            target_n = tf.linalg.einsum("bijc,bijd->bcd", target_f, target_f)/(h*w*c)

            loss += tf.reduce_mean(tf.abs(img_n - target_n)**2)
        return loss

    def recon_criterion(self, input, target, mask_list):
        input = tf.cast(input, tf.float32)
        target = tf.cast(target, tf.float32)
        return 1. * tf.reduce_mean(tf.abs(input - target) * mask_list) + 5. * tf.reduce_mean(tf.abs(input - target) * (1 - mask_list))

    def total_variation_loss(self, image, mask_list):
        def high_pass_x_y(image):
            x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
            y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

            return x_var, y_var
        kernel = tf.ones((3, 3, mask_list.shape[3], mask_list.shape[3]))
        dilated_mask = tf.nn.conv2d(1 - mask_list, kernel, strides=[1, 1, 1, 1], padding="SAME")
        #dilated_ratio = 9. * 3 / (dilated_mask + 10e-6)
        dilated_mask = tf.cast(tf.greater(dilated_mask, 0), "float32")
        image = dilated_mask * image
        x_deltas, y_deltas = high_pass_x_y(image)

        return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

    def vgg_layers(self, layer_names):

        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def get_recon_loss(self,decoder_output, image_list, mask_list):

        comp_list = decoder_output * (1 - mask_list) + image_list * mask_list
        style_loss = (self.compute_style_loss(self.vgg, decoder_output, image_list) + self.compute_style_loss(self.vgg, comp_list, image_list)) / 2.
        perc_loss = (self.compute_vgg_loss(self.vgg, decoder_output, image_list) + self.compute_vgg_loss(self.vgg, comp_list, image_list)) / 2.

        l1_loss = self.recon_criterion(decoder_output, image_list, mask_list)

        tv_loss = self.total_variation_loss(decoder_output, mask_list)
        cross_entroy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l1_loss = 1.* cross_entroy(image_list*mask_list,decoder_output*mask_list) + 5. * cross_entroy(image_list*(1-mask_list),decoder_output*(1-mask_list))


        return l1_loss #+ style_loss * 120. + perc_loss * 0.05 + tv_loss * 0.01

gen_optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001, beta_1= 0.5, beta_2= 0.999)

@tf.function()
def trainer_step(mask_image_list, image_list, mask_list, nvae_trainer):
    image_list = tf.cast(image_list, tf.float32)
    mask_list = tf.cast(mask_list, tf.float32)
    with tf.GradientTape() as gen_tape , tf.GradientTape() as dis_tape:

        decoder_output, kl_losses = nvae_trainer.nvae(mask_image_list)
        decoder_output = tf.cast(decoder_output, tf.float32)
        recon_loss = nvae_trainer.get_recon_loss(decoder_output, image_list, mask_list)
        loss_total = tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_losses)*0.8

    if(loss_total >0):
        gen_grads = gen_tape.gradient(loss_total, nvae_trainer.nvae.trainable_variables)
        #gen_grads = [tf.clip_by_value(grad, -5e+10, 5e+10) for grad in gen_grads]
        gen_optimizer.apply_gradients(zip(gen_grads, nvae_trainer.nvae.trainable_variables))

    return loss_total


