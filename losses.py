
import tensorflow as tf

def kl(mu, log_var):

    loss = -0.5 * tf.reduce_sum(1 + log_var - mu ** 2 - tf.exp(log_var), axis = [1,2,3])
    return tf.reduce_mean(loss, axis = 0)

def kl_2(delta_mu, delta_log_var, mu, log_var):

    var = tf.exp(log_var)
    delta_var = tf.exp(delta_log_var)

    loss = -0.5 * tf.reduce_sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, axis = [1,2,3])
    return tf.reduce_mean(loss, axis = 0)