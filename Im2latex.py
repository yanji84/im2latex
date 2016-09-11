import tensorflow as tf
import numpy as np
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
training_iters = 20000000
display_step = 10
test_step = 10
dropout = 0.75

def setup_summaries():
    loss = tf.Variable(0.)
    tf.scalar_summary("Loss", loss)
    train_acc = tf.Variable(0.)
    tf.scalar_summary("Train Accuracy", train_acc)
    test_acc = tf.Variable(0.)
    tf.scalar_summary("Test Accuracy", test_acc)
    summary_vars = [loss, train_acc, test_acc]
    summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.merge_all_summaries()
    return summary_placeholders, update_ops, summary_op

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
    summary_placeholders, update_ops, summary_op = setup_summaries()

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
    dataLoader = DataLoader("/Users/jiyan/Downloads/train/", image_dim, clasDict, n_classes, n_steps, debug=False)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.train.SummaryWriter("/tmp/im2latex_logs", sess.graph)

        checkpoint = tf.train.get_checkpoint_state("/tmp")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print "successfully loaded checkpoint"

        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y, images, masks = dataLoader.next_train_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           image: images, mask: masks})
            if step % display_step == 0:
                loss,training_generated = sess.run([cost, generated], feed_dict={x: batch_x, y: batch_y,
                                           image: images, mask: masks})
                train_acc = calculate_accuracy(training_generated, batch_x)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + str(loss) + ", training acc=" + str(train_acc))

                test_batch_x, test_batch_y, test_images, test_masks = dataLoader.next_test_batch(batch_size)
                samples = sess.run(generated, feed_dict={x: test_batch_x, y: test_batch_y,
                                                   image: test_images, mask: test_masks})
                test_acc = calculate_accuracy(samples, test_batch_x)
                print("Iter " + str(step*batch_size) + ", Test Accuracy= " + str(test_acc))

                save_path = saver.save(sess, "/tmp/im2latex.ckpt")
                print("Model saved in file: %s" % save_path)

                sess.run(update_ops[0], feed_dict={summary_placeholders[0]:float(loss)})
                sess.run(update_ops[1], feed_dict={summary_placeholders[1]:float(train_acc)})
                sess.run(update_ops[2], feed_dict={summary_placeholders[2]:float(test_acc)})
                summary_str = sess.run(summary_op)
                writer.add_summary(summary_str, step)
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
