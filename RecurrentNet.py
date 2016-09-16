import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from Globals import *
from ConvNet import *
from TransformerNet import *

class RecurrentNet(object):
    def __init__(self, variable_scope, x, y, image, mask):
        self.n_channels = constants['n_channels']
        self.image_dim = constants['image_dim']
        self.n_steps = constants['n_steps']
        self.n_classes = constants['n_classes']
        self.vocabulary_size = constants['vocabulary_size']
        self.embedding_size = constants['embedding_size']
        self.hidden_dim = constants['hidden_dim']
        self.batch_size = constants['batch_size']

        self.x = x
        self.y = y
        self.image = image
        self.mask = mask

        self.stn = SpatialTransformerNetwork("spatial_transformer_network")
        self.convNet = ConvNet("attention_cnn", image, None, None, True)
        self.ctx_convnet = ConvNet("initial_ctx_convnet", self.image, None, None, False)

        with tf.variable_scope(variable_scope):
            self.embeddings = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_size]))
            self.Wx = tf.Variable(tf.random_normal([self.embedding_size, 4*self.hidden_dim]))
            self.Wh = tf.Variable(tf.random_normal([self.hidden_dim, 4*self.hidden_dim]))
            self.Wa = tf.Variable(tf.random_normal([7*7*64, 4*self.hidden_dim]))
            self.b = tf.Variable(tf.zeros(4 * self.hidden_dim))
            self.Wfc = tf.Variable(tf.random_normal([self.hidden_dim, self.n_classes]))
            self.bfc = tf.Variable(tf.zeros(self.n_classes))
            self.intial_context_w = tf.Variable(tf.random_normal([7*7*64, self.hidden_dim]))
            self.intial_context_b = tf.Variable(tf.zeros(self.hidden_dim))

    def lstm_step_forward(self, x_idx, prev_h, prev_c, att):
        Wx = self.Wx
        Wh = self.Wh
        Wa = self.Wa
        b = self.b

        next_h, next_c = None, None

        H = self.hidden_dim

        x = tf.nn.embedding_lookup(self.embeddings, x_idx)

        activation_vector = tf.add(tf.add(tf.matmul(x, Wx), tf.matmul(prev_h, Wh)), tf.add(tf.matmul(att, Wa), b))
        ai = activation_vector[:,0:H]
        af = activation_vector[:,H:2*H]
        ao = activation_vector[:,H*2:3*H]
        ag = activation_vector[:,H*3:4*H]

        i = tf.sigmoid(ai)
        f = tf.sigmoid(af)
        o = tf.sigmoid(ao)
        g = tf.tanh(ag)

        next_c = tf.add(tf.mul(prev_c, f), tf.mul(i, g))
        tanh_cell = tf.tanh(next_c)
        next_h = tf.mul(o, tanh_cell)

        attended = self.stn.localize(next_h, self.image)
        next_att = self.convNet.extract_conv_features(attended)
        next_att = tf.reshape(next_att, [-1, 7*7*64])

        return next_h, next_c, next_att

    def lstm_forward(self):
        generated = []
        generated_logits = []
        features = self.ctx_convnet.extract_conv_features(self.image)
        features = tf.reshape(features, [-1, 7*7*64])
        h0 = tf.matmul(features, self.intial_context_w) + self.intial_context_b
        x = tf.transpose(self.x, perm=[1,0])
        mask = tf.transpose(self.mask, perm=[1,0])
        y = tf.transpose(self.y, perm=[1,0,2])
        hprev = h0
        cprev = 0
        attprev = features
        costs = []
        xt = x[0,:]
        for t in range(self.n_steps):
            generated.append(xt)
            if t < self.n_steps - 1:
                hprev, cprev, attprev = self.lstm_step_forward(x[t,:], hprev, cprev, attprev)
                logits = tf.matmul(hprev, self.Wfc) + self.bfc
                generated_logits.append(logits)
                xt = tf.cast(tf.argmax(logits, 1), tf.int32)
                cost_at_t = tf.nn.softmax_cross_entropy_with_logits(logits, y[t,:,:])
                #batch_cost_at_t = tf.nn.softmax_cross_entropy_with_logits(logits, y[t,:,:])
                #batch_cost_at_t_after_mask = tf.mul(batch_cost_at_t, mask[t,:])
                #avg_cost_at_t = tf.reduce_mean(batch_cost_at_t_after_mask)
                #avg_cost_at_t = tf.reduce_mean(batch_cost_at_t_after_mask)
                costs.append(cost_at_t)

        cost = tf.reduce_mean(tf.reduce_sum(costs, reduction_indices=[0]))        
        generated = tf.pack(generated)
        generated = tf.transpose(generated, perm=[1,0])
        generated_logits = tf.pack(generated_logits)
        generated_logits = tf.transpose(generated_logits, perm=[1,0])
        return cost, generated, generated_logits

    def sample(self):
        generated = []
        features = self.ctx_convnet.extract_conv_features(self.image)
        features = tf.reshape(features, [-1, 7*7*64])
        h0 = tf.matmul(features, self.intial_context_w) + self.intial_context_b
        x = tf.transpose(self.x, perm=[1,0])
        hprev = h0
        cprev = tf.Variable(0,dtype=tf.float32)
        attprev = features
        x = x[0,:]
        for t in range(self.n_steps):
            generated.append(x)
            hprev, cprev, attprev = self.lstm_step_forward(x, hprev, cprev, attprev)
            logits = tf.matmul(hprev, self.Wfc) + self.bfc
            x = tf.cast(tf.argmax(logits, 1), tf.int32)

        generated = tf.pack(generated)
        generated = tf.transpose(generated, perm=[1,0])
        return generated