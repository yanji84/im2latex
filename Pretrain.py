import tensorflow as tf
import numpy as np
import cv2
import math
import random
import os
import sys
from SvhnNet import *

debug = True

whiteWash = True
dataRatio = [0.6,0.3,0.1] # train, test, validate

allLabelsTrain = []
allImagesTrain = []
allLabelsTest = []
allImagesTest = []
allLabelsValidate = []
allImagesValidate = []
validateFiles = []

imageSize = 32
batchSize = 128
trainingIters = 1000000 # in terms of sample size
displayStep = 1 # how often to print details
step = 0

#imagePath = "/Users/jiyan/Desktop/class/"
logPath = "bboxlogs"
modelPath = "bboxModel/"
imagePath = "/Users/jiyan/Downloads/train/"
#imagePath = "/home/deeplearningdev/im2latex/train/"

def setupSummaries():
  with tf.variable_scope('monitor') as scope:
    loss = tf.Variable(0.0)
    tf.scalar_summary("Loss", loss)
    trainAcc = tf.Variable(0.0)
    tf.scalar_summary("Train Accuracy", trainAcc)
    testAcc = tf.Variable(0.0)
    tf.scalar_summary("Test Accuracy", testAcc)
    summaryVars = [loss, trainAcc, testAcc]
    summaryPlaceholders = [tf.placeholder("float") for i in range(len(summaryVars))]
    updateOps = [summaryVars[i].assign(summaryPlaceholders[i]) for i in range(len(summaryVars))]
    return summaryPlaceholders, updateOps

def load():
  global allImagesTrain
  global allLabelsTrain
  global allImagesTest
  global allLabelsTest
  global allImagesValidate
  global allLabelsValidate
  allImages = []
  allLabels = []
  with open("bbox.out") as f:
    content = f.readlines()
    for line in content:
      parts = line.split(",")
      fileName = imagePath + parts[0]
      print fileName
      img = cv2.resize(cv2.imread(fileName, cv2.IMREAD_UNCHANGED), (imageSize, imageSize))
      t,l,w,h = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])

      # white wash image
      if whiteWash:
        imgMean = np.mean(img)
        #std = np.sqrt(np.sum(np.square(img - imgMean)) / (32 * 32))
        img = img.astype(np.float32)
        img -= imgMean
        #img /= std

      allImages.append(img)
      allLabels.append([t,l,w,h])
      if debug and len(allLabels) > 1000:
        break

  trainIdx = int(len(allLabels) * dataRatio[0])
  testIdx = int(trainIdx + len(allLabels) * dataRatio[1])
  allImagesTrain = allImages[:trainIdx]
  allLabelsTrain = allLabels[:trainIdx]
  allImagesTest = allImages[trainIdx:testIdx]
  allLabelsTest = allLabels[trainIdx:testIdx]
  allImagesValidate = allImages[testIdx:]
  allLabelsValidate = allLabels[testIdx:]

def next(size, imgs, labels):
  indices = random.sample(range(len(imgs)), size)
  batchImages = np.array(imgs)[indices]
  batchLabels = np.array(labels)[indices]
  return batchImages,batchLabels

def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def calAccuracy(predBbox, labelBbox):
  from collections import namedtuple
  Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
  totalAcc = 0.0
  for i in range(len(predBbox)):
    ra = Rectangle(predBbox[i][1], predBbox[i][0], predBbox[i][1] + predBbox[i][2], predBbox[i][0] + predBbox[i][3])
    rb = Rectangle(labelBbox[i][1], labelBbox[i][0], labelBbox[i][1] + labelBbox[i][2], labelBbox[i][0] + labelBbox[i][3])
    overlapping = area(ra, rb)
    if overlapping is not None:
      labelArea = labelBbox[i][2] * labelBbox[i][3]
      acc = float(overlapping) / float(labelArea)
      totalAcc += acc
  avgAcc = totalAcc / float(len(predBbox))
  return avgAcc

def train():
  global allImagesValidate
  global allLabelsValidate

  x = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])
  y = tf.placeholder(tf.float32, [None, 4])

  load()
  cnn = SvhnNet(x, y)
  monitorPh, monitorOps = setupSummaries()
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  summaryOps = tf.merge_all_summaries()
  with tf.Session() as sess:
      sess.run(init)
      writer = tf.train.SummaryWriter(logPath, sess.graph)
      if not os.path.exists(modelPath):
        os.makedirs(modelPath)
      checkpoint = tf.train.get_checkpoint_state(modelPath)
      if checkpoint and checkpoint.model_checkpoint_path:
          saver.restore(sess, checkpoint.model_checkpoint_path)
          print "successfully loaded checkpoint"

      step = 1
      while step * batchSize < trainingIters:
          trainImages, trainLabels = next(batchSize, allImagesTrain, allLabelsTrain)
          sess.run(cnn.optimizer, feed_dict={x: trainImages, y: trainLabels})
          if step % displayStep == 0:
              # Calculate training loss and accuracy
              loss, trainBbox = sess.run([cnn.cost,cnn.bbox], feed_dict={x: trainImages,
                                                                            y: trainLabels})
              trainAcc = calAccuracy(trainBbox, trainLabels)
              # calculate test accuracy
              testImages, testLabels = next(batchSize, allImagesTest, allLabelsTest)
              testBbox = sess.run(cnn.bbox, feed_dict={x: testImages, y: testLabels})
              testAcc = calAccuracy(testBbox, testLabels)
              sess.run([monitorOps[0], monitorOps[1], monitorOps[2]], feed_dict={monitorPh[0]:float(loss),
                                                                                 monitorPh[1]:trainAcc,
                                                                                 monitorPh[2]:testAcc})

              print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(trainAcc) + ", Test Accuracy= " + "{:.5f}".format(testAcc))
              
              savePath = saver.save(sess, modelPath + "im2latex.ckpt")
              print("Model saved in file: %s" % savePath)
              summaryStr = sess.run(summaryOps, feed_dict={x: trainImages,
                                                           y: trainLabels})
              writer.add_summary(summaryStr, step)
              writer.add_summary(summaryStr, step)
          step += 1

      print("Optimization Finished!")

if __name__ == '__main__':
  train()



