import tensorflow as tf

class alexnet:
    def __init__(self, imgs, labels):
        with tf.variable_scope("alexnet"):
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
                'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
                'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
                'wd1': tf.Variable(tf.random_normal([4, 4, 256, 1024])),
                'wd2': tf.Variable(tf.random_normal([1024, 1024])),
                'out': tf.Variable(tf.random_normal([1024, 10]))
            }
            biases = {
                'bc1': tf.Variable(tf.random_normal([64])),
                'bc2': tf.Variable(tf.random_normal([128])),
                'bc3': tf.Variable(tf.random_normal([256])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'bd2': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

        dropout = 0.85
        imgs = tf.reshape(imgs, shape=[-1, 28, 28, 1])

        conv1 = conv2d('conv1', imgs, weights['wc1'], biases['bc1'])
        pool1 = max_pool('pool1', conv1, k=2)
        norm1 = norm('norm1', pool1, lsize=4)
        norm1 = tf.nn.dropout(norm1, dropout)

        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        pool2 = max_pool('pool2', conv2, k=2)
        norm2 = norm('norm2', pool2, lsize=4)
        norm2 = tf.nn.dropout(norm2, dropout)

        conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        pool3 = max_pool('pool3', conv3, k=2)
        norm3 = norm('norm3', pool3, lsize=4)
        norm3 = tf.nn.dropout(norm3, dropout)

        dense1 = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]]) 
        dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1') 
        dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2') # Relu activation
        self.logits = tf.matmul(dense2, weights['out']) + biases['out']
        self.labels = labels

    def forward(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        
        self.correct_pred = tf.equal(tf.argmax(self.fc3l, 1), tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def conv2d(name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

    def max_pool(name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def norm(name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)