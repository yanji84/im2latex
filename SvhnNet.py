import tensorflow as tf

class SvhnNet:
    def __init__(self, imgs, labels):
      #imgs = tf.image.random_brightness(imgs,max_delta=63)
      #imgs = tf.image.random_contrast(imgs,lower=0.2, upper=1.8)
      tf.image_summary('images', imgs, max_images = 3)

      # conv1
      with tf.variable_scope('conv1') as scope:
        kernel = self.variable_with_weight_decay('weights', [5, 5, 3, 64],
                                             1e-4, 0.0)
        conv = tf.nn.conv2d(imgs, kernel, [1, 1, 1, 1], padding='SAME')
        biases = self.variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        self.activation_summary(conv1)

        with tf.variable_scope('visualization'):
            x_min = tf.reduce_min(kernel)
            x_max = tf.reduce_max(kernel)
            kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
            kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
            tf.image_summary('conv1/filters', kernel_transposed, max_images=64)

      # pool1
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

      # norm1
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

      # conv2
      with tf.variable_scope('conv2') as scope:
        kernel = self.variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = self.variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        self.activation_summary(conv2)
    
      # norm2
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')

      # pool2
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')

      # local3
      with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
          dim *= d
        reshape = tf.reshape(pool2, [-1, dim])

        weights = self.variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = self.variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        self.activation_summary(local3)

      # local4
      with tf.variable_scope('local4') as scope:
        weights = self.variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = self.variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)
        self.activation_summary(local4)

      # softmax, i.e. softmax(WX + b)
      with tf.variable_scope('final') as scope:
        weights1 = self.variable_with_weight_decay('weights_local4', [192, 4],
                                              stddev=1/192.0, wd=0.0)
        weights2 = self.variable_with_weight_decay('weights_conv2', [dim, 4],
                                              stddev=1/dim.0, wd=0.0)
        biases = self.variable_on_cpu('biases', [4],
                                  tf.constant_initializer(0.0))
        
        self.bbox = tf.matmul(local4, weights1) + tf.matmul(reshape, weights2) + biases

        self.activation_summary(self.bbox)

      with tf.variable_scope('results') as scope:
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - self.bbox), [1]), name='mean_square_error')

        tf.add_to_collection('losses', loss)

        self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

    def variable_on_cpu(self, name, shape, initializer):
      with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
      return var

    def variable_with_weight_decay(self, name, shape, stddev, wd):
      var = self.variable_on_cpu(name, shape,
                            tf.truncated_normal_initializer(stddev=stddev))
      if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
      return var

    def activation_summary(self, x):
      tf.histogram_summary(x.op.name + '/activations', x)
      tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))