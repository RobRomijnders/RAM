# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:54:35 2016

@author: rob
"""

import tensorflow as tf
#import tf_mnist_loader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import sys
import socket
if 'rob-laptop' in socket.gethostname():
  data_directory = '/home/rob/Dropbox/ConvNets/tf/MNIST'
  sys.path.append('/home/rob/Dropbox/ml_projects/RAM/')
elif 'rob-com' in socket.gethostname():
  data_directory = '/home/rob/Documents/RAM/MNIST'
  sys.path.append('/home/rob/Documents/RAM/')
  save_dir = '/home/rob/Documents/RAM/canvas/'


#dataset = tf_mnist_loader.read_data_sets("mnist_data")
from tensorflow.examples.tutorials import mnist
from model import Model
dataset = mnist.input_data.read_data_sets(data_directory, one_hot=False)
start_step = 0
max_iters = 1000000
config = {}
config['batch_size'] = batch_size = 10
config['glimpses'] = glimpses = 6
config['depth'] = depth = 3
config['sensorBandwidth'] = sensorBandwidth = 8
config['min_radius'] = min_radius = 2 # zooms -> min_radius * 2**<depth_level>
config['max_radius'] = max_radius = min_radius * (2 ** (depth - 1))




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
  reward_ma = 0.0
  cost_ma = 0.0
  for step in xrange(start_step + 1, max_iters):
    nextX, nextY = dataset.train.next_batch(batch_size)
    feed_dict = {model.image: nextX, model.labels: nextY}
    fetches = [model.train_op, model.cost, model.reward, model.labels_pred, model.glimpse_images, model.locs]

    results = sess.run(fetches, feed_dict=feed_dict)
    _, cost_fetched, reward_fetched, prediction_labels_fetched, f_glimpse_images_fetched, sampled_locs_fetched = results

    #Some moving averages
    reward_ma = 0.9*reward_ma + 0.1*reward_fetched
    cost_ma = 0.9*cost_ma + 0.1*cost_fetched
    if step % 500 == 0:
      if step % 5000 == 0:
        evaluate()
      if draw and animate:
        plt.close('all')
        model.draw_ram(f_glimpse_images_fetched,prediction_labels_fetched,sampled_locs_fetched,nextX,nextY,save_dir='/home/rob/Documents/RAM/canvas/')
      print('Step %6.0f: cost = %6.2f(%6.2f) reward = %4.1f(%4.2f) ' % (step, cost_fetched,cost_ma, reward_fetched,reward_ma))

      summary_str = sess.run(model.summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)

#Now go to the directory and run  (after install ImageMagick)
#  convert -delay 20 -loop 0 *.png RAM.gif

sess.close()