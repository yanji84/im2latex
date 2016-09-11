import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tflearn.datasets.mnist as mnist
from Globals import *
from ConvNet import *
from DataLoader import *
from RecurrentNet import *

n_channels = constants['n_channels']
image_dim = constants['image_dim']
n_steps = gl.constants['n_steps']
n_classes = constants['n_classes']
learning_rate = constants['learning_rate']
batch_size = constants['batch_size']

# Parameters
training_iters = 200000
display_step = 10
test_step = 10
dropout = 0.75

def calculate_accuracy(samples, batch_x):
    step = 0.0
    correct = 0.0
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            if j != 0:
                if samples[i][j] == batch_x[i][j]:
                    correct += 1.0
                step += 1.0
                if samples[i][j] == 11:
                    break
    return correct / step

# this is the final model
def train_digit_sequence():
    x = tf.placeholder(shape = (None, n_steps), dtype=tf.int32)
    y = tf.placeholder(shape = (None, n_steps - 1, n_classes), dtype=tf.float32)
    mask = tf.placeholder(shape = (None, n_steps), dtype=tf.float32)
    image= tf.placeholder(shape = (None, image_dim, image_dim, n_channels), dtype=tf.float32)
    rnn = RecurrentNet("recurrent_net", x, y , image, mask)
    cost,generated = rnn.lstm_forward()
    generated = rnn.sample()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    clasDict = {}
    clasDict['#'] = 0
    clasDict['0'] = 1
    clasDict['1'] = 2
    clasDict['2'] = 3
    clasDict['3'] = 4
    clasDict['4'] = 5
    clasDict['5'] = 6
    clasDict['6'] = 7
    clasDict['7'] = 8
    clasDict['8'] = 9
    clasDict['9'] = 10
    clasDict['*'] = 11
    dataLoader = DataLoader("/Users/jiyan/Downloads/train/", image_dim, clasDict, n_classes, n_steps, debug=True)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y, images, masks = dataLoader.next_train_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           image: images, mask: masks})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss,training_generated = sess.run([cost, generated], feed_dict={x: batch_x, y: batch_y,
                                           image: images, mask: masks})
                acc = calculate_accuracy(training_generated, batch_x)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + str(loss) + ", training acc=" + str(acc))

            if step % test_step == 0:
                batch_x, batch_y, images, masks = dataLoader.next_test_batch(batch_size)
                samples = sess.run(generated, feed_dict={x: batch_x, y: batch_y,
                                                   image: images, mask: masks})
                acc = calculate_accuracy(samples, batch_x)
                print("Iter " + str(step*batch_size) + ", Test Accuracy= " + str(acc))
            step += 1
        print("Optimization Finished!")

train_digit_sequence()

# tests
#train_mnist()

'''
import numpy as np
clasDict = {}
clasDict['#'] = 0
clasDict['0'] = 1
clasDict['1'] = 2
clasDict['2'] = 3
clasDict['3'] = 4
clasDict['4'] = 5
clasDict['5'] = 6
clasDict['6'] = 7
clasDict['7'] = 8
clasDict['8'] = 9
clasDict['9'] = 10
clasDict['*'] = 11
dataLoader = DataLoader("/Users/jiyan/Downloads/train/", image_dim, clasDict, n_classes, n_steps, debug=True)
x,y,img,masks = dataLoader.next_train_batch(3)
print x
x,y,img,masks = dataLoader.next_test_batch(3)
print x
'''

'''
x = tf.placeholder(shape = (None, n_steps), dtype=tf.int32)
y = tf.placeholder(shape = (None, n_steps - 1, n_classes), dtype=tf.float32)
mask = tf.placeholder(shape = (None, n_steps), dtype=tf.float32)
image= tf.placeholder(shape = (None, image_dim, image_dim, n_channels), dtype=tf.float32)
rnn = RecurrentNet("recurrent_net", x, y , image, mask)
cost = rnn.lstm_forward()
generated = rnn.sample()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

clasDict = {}
clasDict['#'] = 0
clasDict['0'] = 1
clasDict['1'] = 2
clasDict['2'] = 3
clasDict['3'] = 4
clasDict['4'] = 5
clasDict['5'] = 6
clasDict['6'] = 7
clasDict['7'] = 8
clasDict['8'] = 9
clasDict['9'] = 10
clasDict['*'] = 11
dataLoader = DataLoader("/Users/jiyan/Downloads/train/", image_dim, clasDict, n_classes, n_steps, debug=True)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    batch_x, batch_y, images, masks = dataLoader.next_test_batch(batch_size)
    samples = sess.run(generated, feed_dict={x: batch_x, y: batch_y,
                                       image: images, mask: masks})
    print samples
    print batch_x
    print calculate_accuracy(samples, batch_x)
'''
