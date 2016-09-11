import tensorflow as tf
import numpy as np
from Globals import *
from SpatialTransformer import transformer

class SpatialTransformerNetwork(object):
    def __init__(self, variable_scope):
        with tf.variable_scope(variable_scope):
            self.n_theta = 6
            self.out_size = constants['stn_out_size']
            self.hidden_dim = constants['hidden_dim']
            self.Wx = tf.Variable(tf.zeros([self.hidden_dim, self.n_theta]))

            initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()

            self.b = tf.Variable(initial_value=initial)

    def localize(self, x, image):
        theta = tf.matmul(x, self.Wx) + self.b
        h_trans = transformer(image, theta, self.out_size)
        return h_trans