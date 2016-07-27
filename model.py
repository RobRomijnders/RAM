# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:28:57 2016

@author: rob
"""

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


class Model():
  def __init__(self,config):
    self.depth = config['depth'] # zooms
    min_radius = config['min_radius']
    max_radius = config['max_radius']
    self.sensorBandwidth = config['sensorBandwidth'] # fixed resolution of sensor
    sensorArea = self.sensorBandwidth**2
    channels = 1 # grayscale
    totalSensorBandwidth = self.depth * self.sensorBandwidth * self.sensorBandwidth * channels
    self.batch_size = config['batch_size']

    hg_size = 128
    hl_size = 128
    g_size = 256
    cell_size = 256
    cell_out_size = cell_size

    self.glimpses = config['glimpses']
    n_classes = 10

    lr = 1e-4

    mnist_size = 28

    loc_sd = 0.1
    mean_locs = []
    self.sampled_locs = [] # ~N(mean_locs[.], loc_sd)
    self.glimpse_images = [] # to show in window


    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=1.0/shape[0]) # for now
      return tf.Variable(initial)


    def linear(x,output_dim):
      """ Function to compute linear transforms
      when x is in [batch_size, some_dim] then output will be in [batch_size, output_dim]
      Be sure to use right variable scope n calling this function
      """
      w=tf.get_variable("w", [x.get_shape()[1], output_dim])
      b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
      return tf.nn.xw_plus_b(x,w,b)


    def sensor_glimpse(img, normLoc):
      loc = ((normLoc + 1.0) / 2.0) * mnist_size # normLoc coordinates are between -1 and 1
      loc = tf.cast(loc, tf.int32)

      #Random perturbations cannot exceed the image
      loc = tf.clip_by_value(loc, 0, 28)

      img = tf.reshape(img, (self.batch_size, mnist_size, mnist_size, channels))

      zooms = []

      # process each image individually
      for k in xrange(self.batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        offset = max_radius

        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
            max_radius * 2 + mnist_size, max_radius * 2 + mnist_size)

        for i in xrange(self.depth):
          r = int(min_radius * (2 ** (i-1)))

          d_raw = 2 * r
          d = tf.constant(d_raw, shape=[1])

          d = tf.tile(d, [2])

          loc_k = loc[k,:]
          adjusted_loc = offset + loc_k - r


          one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,\
              one_img.get_shape()[1].value))
          # crop image to (d x d)
          zoom = tf.slice(one_img2, adjusted_loc, d, name='crop_image')

          # resize cropped image to (sensorBandwidth x sensorBandwidth)
          zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (self.sensorBandwidth, self.sensorBandwidth))
          zoom = tf.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
          imgZooms.append(zoom)

        zooms.append(tf.pack(imgZooms))

      zooms = tf.pack(zooms)

      self.glimpse_images.append(zooms)

      return zooms

    def get_glimpse(loc):
      glimpse_input = sensor_glimpse(self.image, loc)

      glimpse_input = tf.reshape(glimpse_input, (self.batch_size, totalSensorBandwidth))

      l_hl = weight_variable((2, hl_size))
      l_hl_bias = tf.Variable(tf.constant(0.1, shape=[hl_size]))
      glimpse_hg = weight_variable((totalSensorBandwidth, hg_size))
      glimpse_hg_bias = tf.Variable(tf.constant(0.1,shape=[hg_size]))

      hg = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, glimpse_hg,glimpse_hg_bias))
      hl = tf.nn.relu(tf.nn.xw_plus_b(loc, l_hl,l_hl_bias))

      hg_g = weight_variable((hg_size, g_size))
      hg_g_bias = tf.Variable(tf.constant(0.1,shape=[g_size]))
      hl_g = weight_variable((hl_size, g_size))
      hl_g_bias = tf.Variable(tf.constant(0.1,shape=[g_size]))

      g = tf.nn.relu(tf.nn.xw_plus_b(hg, hg_g,hg_g_bias) + tf.nn.xw_plus_b(hl, hl_g,hl_g_bias))

      return g

    def get_next_input(output, i):
      mean_loc = tf.tanh(tf.nn.xw_plus_b(output, h_l_out,b_l_out))  # in[batch_size, 2]
      mean_locs.append(mean_loc)

      sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)



      self.sampled_locs.append(sample_loc)

      return get_glimpse(sample_loc)

    # to use for maximum likelihood with glimpse location
    def gaussian_pdf(mean, sample):
      Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
      a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
      return Z * tf.exp(a)

    self.image = tf.placeholder(tf.float32, shape=(self.batch_size, 28 * 28), name="images")
    self.labels = tf.placeholder(tf.int64, shape=(self.batch_size), name="labels")

    h_l_out = weight_variable((cell_out_size, 2))
    b_l_out = tf.Variable(tf.constant(0.1,shape=[2]))
    loc_mean = weight_variable((self.batch_size, self.glimpses, 2))

    initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)

    initial_glimpse = get_glimpse(initial_loc)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size, g_size, num_proj=cell_out_size)
    initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

    inputs = [initial_glimpse]
    inputs.extend([0] * (self.glimpses - 1))

    outputs, _ = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=get_next_input)
    get_next_input(outputs[-1], 0)


    # convert list of tensors to one big tensor
    self.sampled_locs = tf.concat(0, self.sampled_locs)
   # self.sampled_locs = tf.reshape(self.sampled_locs, (self.batch_size, self.glimpses, 2))
    self.sampled_locs = tf.reshape(self.sampled_locs, (self.glimpses, self.batch_size,2))
    self.sampled_locs = tf.transpose(self.sampled_locs, [1,0,2])
    mean_locs = tf.concat(0, mean_locs)
    #mean_locs = tf.reshape(mean_locs, (self.batch_size, self.glimpses, 2))
    mean_locs = tf.reshape(mean_locs, (self.glimpses, self.batch_size, 2))
    mean_locs = tf.transpose(mean_locs, [1,0,2])
    self.glimpse_images = tf.concat(0, self.glimpse_images)


    outputs = outputs[-1] # look at ONLY THE END of the sequence
    outputs = tf.reshape(outputs, (self.batch_size, cell_out_size))
    with tf.variable_scope('classification'):
      a_y = linear(outputs,n_classes)
    cost_sm = tf.nn.sparse_softmax_cross_entropy_with_logits(a_y, self.labels)
    self.labels_pred = tf.arg_max(a_y, 1)

    R = tf.cast(tf.equal(self.labels_pred, self.labels), tf.float32) # reward per example

    self.reward = tf.reduce_mean(R) # overall reward

    p_loc = gaussian_pdf(mean_locs, self.sampled_locs)
    p_loc = tf.reshape(p_loc, (self.batch_size, self.glimpses * 2))

    R = tf.reshape(R, (self.batch_size, 1))
    J = tf.log(p_loc + 1e-9) * R
    J = tf.reduce_sum(J, 1) - cost_sm
    J = tf.reduce_mean(J, 0)
    self.cost = -J

    global_step = tf.Variable(0,trainable=False)
    lrate = tf.train.exponential_decay(lr,global_step,1000,0.5,staircase=True)

    optimizer = tf.train.AdamOptimizer(lrate)
    self.train_op = optimizer.minimize(self.cost)


    tf.scalar_summary("reward", self.reward)
    tf.scalar_summary("cost", self.cost)

    self.summary_op = tf.merge_all_summaries()
    prefix = tf.expand_dims(initial_loc,1)
    self.locs = tf.concat(1,[prefix,self.sampled_locs])

  def draw_ram(self,f_glimpse_images_fetched,prediction_labels_fetched,sampled_locs_fetched,nextX,nextY,save_dir=None):
    fig = plt.figure()
    txt = fig.suptitle("-", fontsize=36, fontweight='bold')
    plt.ion()
    plt.show()
    plt.subplots_adjust(top=0.7)
    plotImgs = []

    f_glimpse_images = np.reshape(f_glimpse_images_fetched, (self.glimpses + 1, self.batch_size, self.depth, self.sensorBandwidth, self.sensorBandwidth)) #steps, THEN batch
    fillList = False

    if len(plotImgs) == 0:
      fillList = True

    # display first in mini-batch
    for y in xrange(self.glimpses):
      txt.set_text('FINAL PREDICTION: %i\nTRUTH: %i\nSTEP: %i/%i'
          % (prediction_labels_fetched[0], nextY[0], (y + 1), self.glimpses))

      for x in xrange(self.depth):
        plt.subplot(self.depth, 2, x + 1)
        if fillList:
          plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'), interpolation="nearest")
          plotImg.autoscale()
          plotImgs.append(plotImg)
        else:
          plotImgs[x].set_data(f_glimpse_images[y, 0, x])
          plotImgs[x].autoscale()

      fillList = False

      ax = fig.add_subplot(324)
      ax.imshow(np.reshape(nextX[0],(28,28)), cmap=plt.get_cmap('gray'))
#      ax.add_patch(patches.Rectangle(loc,5,5,fill=False,linestyle='solid',color='r'))
      loc = ((sampled_locs_fetched[0,y,:]+1)*14).astype(int)
      ax.add_patch(patches.Rectangle(np.flipud(loc)-np.array([2,2]),4,4,fill=False,linestyle='solid',color='r'))

      fig.canvas.draw()
      fig.subplots_adjust(hspace=0)  #No horizontal space between subplots
      fig.subplots_adjust(wspace=0)  #No vertical space between subplots
      if save_dir is not None:
        plt.savefig(save_dir+'canvas'+str(nextY[0])+str(10+y)+'.png')
      time.sleep(0.1)
      plt.pause(0.0001)
    #ax.remove()
    return