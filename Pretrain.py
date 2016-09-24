import tensorflow as tf
import numpy as np
import cv2
import math
from vgg16 import *
from alexnet import *

debug = True
batch_size = 128
training_iters = 1000000
onehotLabels = None
allLabels = []
allImages = []
display_step = 1
idx = 0
step = 0

imagePath = "/Users/jiyan/Desktop/class/"
#imagePath = "/home/deeplearningdev/class/"

def load():
  global onehotLabels
  with open("digit.out") as f:
    content = f.readlines()
    for line in content:
      parts = line.split(",")
      fileName = imagePath + parts[0]
      print fileName
      img = cv2.resize(cv2.imread(fileName, cv2.IMREAD_UNCHANGED), (32, 32))

      # white wash image
      mn = np.mean(img)
      y = img.astype(np.float32)
      y -= mn
      img_afterww = y

      allImages.append(img_afterww)
      allLabels.append(parts[1])
      if debug and len(allLabels) > 10:
        break

  labels = np.array(allLabels)
  onehotLabels = np.zeros((len(allLabels), 10))
  onehotLabels[np.arange(len(allLabels)), allLabels] = 1

def next(size):
  global idx
  print idx
  startIdx = idx
  endIdx = idx + size

  if endIdx >= len(allLabels):
      endIdx = len(allLabels) - 1

  images = allImages[startIdx:endIdx]
  ys = onehotLabels[startIdx:endIdx]
  idx = endIdx
  if idx >= len(allLabels) - 1:
      idx = 0
  return images,ys

if __name__ == '__main__':
  x = tf.placeholder(tf.float32, [None, 32, 32, 3])
  y = tf.placeholder(tf.float32, [None, 10])

  load()
  #vgg = vgg16(x, y)
  cnn = alexnet(x, y)
  
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  summary_op = tf.merge_all_summaries()
  with tf.Session() as sess:
      sess.run(init)
      writer = tf.train.SummaryWriter("vgglogs", sess.graph)

      checkpoint = tf.train.get_checkpoint_state("vggmodel/")
      if checkpoint and checkpoint.model_checkpoint_path:
          saver.restore(sess, checkpoint.model_checkpoint_path)
          print "successfully loaded checkpoint"

      step = 1
      # Keep training until reach max iterations
      while step * batch_size < training_iters:
          images, ys = next(batch_size)
          # Run optimization op (backprop)
          sess.run(cnn.optimizer, feed_dict={x: images, y: ys})
          if step % display_step == 0:
              # Calculate batch loss and accuracy
              loss, acc, summary_str = sess.run([cnn.cost,cnn.accuracy,summary_op], feed_dict={x: images,
                                                                y: ys})
              print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
              
              save_path = saver.save(sess, "vggmodel/im2latex.ckpt")
              print("Model saved in file: %s" % save_path)
              writer.add_summary(summary_str, step)
          step += 1
      print("Optimization Finished!")
