import tensorflow as tf
import numpy as np
import Globals as gl

class ConvNet(object):
    def __init__(self, variable_scope, image, y, dropout, smallNetwork):
        self.n_classes = gl.constants['n_classes']
        self.x = image
        self.y = y
        self.smallNetwork = smallNetwork
        self.dropout = dropout

        with tf.variable_scope(variable_scope):
            self.weights = {
                'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64])),
                'wc4': tf.Variable(tf.random_normal([5, 5, 64, 64])),
                'wfc1': tf.Variable(tf.random_normal([7*7*64, 1024])),
                'wfc2': tf.Variable(tf.random_normal([1024, self.n_classes])),
            }

            self.biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bc3': tf.Variable(tf.random_normal([64])),
                'bc4': tf.Variable(tf.random_normal([64])),
                'bfc1': tf.Variable(tf.random_normal([1024])),
                'bfc2': tf.Variable(tf.random_normal([self.n_classes])),
            }

    # Conv2D wrapper, with bias and relu activation
    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # MaxPool2D wrapper
    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def forward(self):
        weights = self.weights
        biases = self.biases
        features = self.extract_conv_features(self.x)

        fc1 = tf.reshape(features, [-1, weights['wfc1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.dropout)
        pred = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.y))
        return cost, pred

    # Return features
    def extract_conv_features(self, images):
        x = images
        weights = self.weights
        biases = self.biases

        # Reshape input picture
        if self.smallNetwork:
            x = tf.reshape(x, shape=[-1, 28,28,3])
        else:
            x = tf.reshape(x, shape=[-1, 112,112,3])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        if self.smallNetwork:
            features = conv2
        else:
            # Convolution Layer
            conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
            # Max Pooling (down-sampling)
            conv3 = self.maxpool2d(conv3, k=2)

            # Convolution Layer
            conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'])
            # Max Pooling (down-sampling)
            conv4 = self.maxpool2d(conv4, k=2)
            features = conv4

        return features