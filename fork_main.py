# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:54:35 2016

@author: rob
"""

import tensorflow as tf
#import tf_mnist_loader
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import socket
if 'rob-laptop' in socket.gethostname():
  data_directory = '/home/rob/Dropbox/ConvNets/tf/MNIST'
  sys.path.append('/home/rob/Dropbox/ml_projects/RAM/')
elif 'rob-com' in socket.gethostname():
  data_directory = '/home/rob/Documents/RAM/MNIST'


#dataset = tf_mnist_loader.read_data_sets("mnist_data")
from tensorflow.examples.tutorials import mnist
from model import Model
dataset = mnist.input_data.read_data_sets(data_directory, one_hot=False)
save_dir = "save-3scales/"
save_prefix = "save"
start_step = 10000
load_path = None
#load_path = save_dir + save_prefix + str(start_step) + ".ckpt"
max_iters = 1000000
config = {}
config['batch_size'] = batch_size = 10
config['glimpses'] = glimpses = 6
config['depth'] = depth = 3
config['sensorBandwidth'] = sensorBandwidth = 8




# to enable visualization, set draw to True
eval_only = False
animate = True
draw = True


model = Model(config)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

def evaluate():
  data = dataset.test
  batches_in_epoch = len(data._images) // batch_size
  accuracy = 0

  for i in xrange(batches_in_epoch):
      nextX, nextY = dataset.test.next_batch(batch_size)
      feed_dict = {model.image: nextX, model.labels: nextY}
      r = sess.run(model.reward, feed_dict=feed_dict)
      accuracy += r

  accuracy /= batches_in_epoch

  print("ACCURACY: " + str(accuracy))

if eval_only:
  evaluate()
else:
  summary_writer = tf.train.SummaryWriter("summary")

  if draw:
    fig = plt.figure()
    txt = fig.suptitle("-", fontsize=36, fontweight='bold')
    plt.ion()
    plt.show()
    plt.subplots_adjust(top=0.7)
    plotImgs = []
  reward_ma = 0.0
  cost_ma = 0.0
  for step in xrange(start_step + 1, max_iters):
    nextX, nextY = dataset.train.next_batch(batch_size)
    feed_dict = {model.image: nextX, model.labels: nextY}
    fetches = [model.train_op, model.cost, model.reward, model.labels_pred, model.glimpse_images]

    results = sess.run(fetches, feed_dict=feed_dict)
    _, cost_fetched, reward_fetched, prediction_labels_fetched, f_glimpse_images_fetched = results

    if step % 20 == 0:
      if step % 1000 == 0:
          #saver.save(sess, save_dir + save_prefix + str(step) + ".ckpt")
          if step % 5000 == 0:
              evaluate()


      ##### DRAW WINDOW ################

      f_glimpse_images = np.reshape(f_glimpse_images_fetched, (glimpses + 1, batch_size, depth, sensorBandwidth, sensorBandwidth)) #steps, THEN batch

      if draw:
        if animate:
          fillList = False
          if len(plotImgs) == 0:
            fillList = True

          # display first in mini-batch
          for y in xrange(glimpses):
            txt.set_text('FINAL PREDICTION: %i\nTRUTH: %i\nSTEP: %i/%i'
                % (prediction_labels_fetched[0], nextY[0], (y + 1), glimpses))

            for x in xrange(depth):
              plt.subplot(depth, 1, x + 1)
              if fillList:
                plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                     interpolation="nearest")
                plotImg.autoscale()
                plotImgs.append(plotImg)
              else:
                plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                plotImgs[x].autoscale()

            fillList = False

            fig.canvas.draw()
            time.sleep(0.1)
            plt.pause(0.0001)
        else:
          txt.set_text('PREDICTION: %i\nTRUTH: %i' % (prediction_labels_fetched[0], nextY[0]))
          for x in xrange(depth):
            for y in xrange(glimpses):
              plt.subplot(depth, glimpses, x * glimpses + y + 1)
              plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                           interpolation="nearest")

          plt.draw()
          time.sleep(0.05)
          plt.pause(0.0001)

      ################################
      reward_ma = 0.8*reward_ma + 0.2*reward_fetched
      cost_ma = 0.8*cost_ma + 0.2*cost_fetched
      print('Step %d: cost = %5.2f(%5.2f) reward = %4.2f(%4.2f) ' % (step, cost_fetched,cost_ma, reward_fetched,reward_ma))

      summary_str = sess.run(model.summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)

sess.close()