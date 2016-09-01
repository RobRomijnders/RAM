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
    self.sensorBW = config['sensorBandwidth'] # fixed resolution of sensor
    sensorArea = self.sensorBW**2
    channels = 1 # grayscale
    BW = self.depth * self.sensorBW * self.sensorBW * channels  #Bandwidth
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

    loc_sd_final = config['loc_sd_final']
    loc_sd_start = config['loc_sd_start']
    global_step = tf.Variable(0,trainable=False)
    loc_sd = tf.train.exponential_decay(loc_sd_start-loc_sd_final, global_step, 1000, 0.8)+ loc_sd_final


    mean_locs = []
    self.sampled_locs = [] # ~N(mean_locs[.], loc_sd)
    self.glimpse_images = [] # to show in window




    def weight_variable(shape,name=None):
      initial = tf.truncated_normal(shape, stddev=1.0/shape[0]) # for now
      return tf.Variable(initial,name=name)


    #Initialize parameters
    #naming according to
    # - W is a trainable weight-matrix
    # - b is a trainable bias-vector
    # l postfixes for LOCATION
    # g postfixes for GLIMPSE
    Wl1 = weight_variable((2, hl_size),name='Wl1')
    bl1 = tf.Variable(tf.constant(0.1, shape=[hl_size]),name='bl1')
    Wg1 = weight_variable((BW, hg_size),name='Wg1')
    bg1 = tf.Variable(tf.constant(0.1,shape=[hg_size]),name='bg1')
    Wg2 = weight_variable((hg_size, g_size),name='Wg2')
    bg2 = tf.Variable(tf.constant(0.1,shape=[g_size]),name='bg2')
    Wl2 = weight_variable((hl_size, g_size),name='Wl2')
    bl2 = tf.Variable(tf.constant(0.1,shape=[g_size]),name='bl2')
    Wl_out = weight_variable((cell_out_size, 2),name='Wl_out')
    bl_out = tf.Variable(tf.constant(0.1,shape=[2]),name='bl_out')


    def sensor_glimpse(img, normLoc):
      loc = ((normLoc + 1.0) / 2.0) * mnist_size
      loc = tf.cast(loc, tf.int32)

      #Random perturbations cannot exceed the image
      loc = tf.clip_by_value(loc, 0, 28)

      img = tf.reshape(img, (self.batch_size, mnist_size, mnist_size, channels))

      zooms = []

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

          # resize cropped image to (sensorBW x sensorBW)
          zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)), (self.sensorBW, self.sensorBW))
          zoom = tf.reshape(zoom, (self.sensorBW, self.sensorBW))
          imgZooms.append(zoom)

        zooms.append(tf.pack(imgZooms))

      zooms = tf.pack(zooms)
      zooms = tf.stop_gradient(zooms)

      self.glimpse_images.append(zooms)

      return zooms

    def get_glimpse(loc):
      glimpse_input = sensor_glimpse(self.image, loc)
      print(glimpse_input)
      glimpse_input = tf.reshape(glimpse_input, (self.batch_size, BW))
      hg = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, Wg1,bg1))
      hl = tf.nn.relu(tf.nn.xw_plus_b(loc, Wl1,bl1))
      g = tf.nn.relu(tf.nn.xw_plus_b(hg, Wg2,bg2) + tf.nn.xw_plus_b(hl, Wl2,bl2))
      return g

    def get_next_input(output, i):
      mean_loc = tf.tanh(tf.nn.xw_plus_b(output, Wl_out,bl_out))
      mean_locs.append(mean_loc)

      sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)



      self.sampled_locs.append(sample_loc)

      return get_glimpse(sample_loc)

    def gaussian_pdf(mean, sample):
      Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
      a = -tf.square(sample - mean) / (2.0 * tf.square(loc_sd))
      return Z * tf.exp(a)

    self.image = tf.placeholder(tf.float32, shape=(self.batch_size, 28 * 28), name="images")
    self.labels = tf.placeholder(tf.int64, shape=(self.batch_size), name="labels")
    self.keep_prob = tf.placeholder("float", name = 'Drop_out')



    initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)

    initial_glimpse = get_glimpse(initial_loc)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size, g_size)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
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
    Wpred_out = weight_variable((cell_out_size, n_classes),name='Wpred_out')
    bpred_out = tf.Variable(tf.constant(0.1, shape=[n_classes]),name='bpred_out')

    a_y = tf.nn.xw_plus_b(outputs,Wpred_out,bpred_out)
    cost_sm = tf.nn.sparse_softmax_cross_entropy_with_logits(a_y, self.labels)
    self.labels_pred = tf.arg_max(a_y, 1)

    R = tf.cast(tf.equal(self.labels_pred, self.labels), tf.float32) # reward per example

    self.reward = tf.reduce_mean(R) # overall reward

    p_loc = gaussian_pdf(mean_locs, self.sampled_locs)
    p_loc = tf.reshape(p_loc, (self.batch_size, self.glimpses * 2))

    R = tf.reshape(R, (self.batch_size, 1))
    J = tf.log(p_loc + 1e-9) * R
    J = tf.reduce_sum(J, 1)
    tvars = tf.trainable_variables()
    self.grads = tf.gradients(J,tvars)
    grads_zip = zip(self.grads,tvars)
    for g,variable in grads_zip:
      print(variable.name)
    J_comb = J  - cost_sm
    J_comb = tf.reduce_mean(J_comb, 0)
    self.cost = -J_comb

    lrate = tf.train.exponential_decay(lr,global_step,10000,0.9,staircase=False)

    optimizer = tf.train.AdamOptimizer(lrate)
    self.train_op = optimizer.minimize(self.cost, global_step=global_step)


    tf.scalar_summary("reward", self.reward)
    tf.scalar_summary("cost", self.cost)

    self.summary_op = tf.merge_all_summaries()
    prefix = tf.expand_dims(initial_loc,1)
    self.locs = tf.concat(1,[prefix,self.sampled_locs])

  def draw_ram(self,f_glimpse_images_fetched,prediction_labels_fetched,sampled_locs_fetched,nextX,nextY,save_dir=None):
    colors = ['r','g']
    fig = plt.figure()
#    txt = fig.suptitle("-", fontsize=36, fontweight='bold')
    plt.show()
    plt.subplots_adjust(top=0.7)
    plotImgs = []

    f_glimpse_images = np.reshape(f_glimpse_images_fetched, (self.glimpses + 1, self.batch_size, self.depth, self.sensorBW, self.sensorBW)) #steps, THEN batch
    fillList = False

    if len(plotImgs) == 0:
      fillList = True

    # display first in mini-batch
    for y in xrange(self.glimpses):
      eq = int(prediction_labels_fetched[0] ==  nextY[0])
      for x in xrange(self.depth):
        plt.subplot(1, 4, x + 1)
        if fillList:
          plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'), interpolation="nearest")
          plotImg.autoscale()
          plotImg.axes.get_xaxis().set_visible(False)
          plotImg.axes.get_yaxis().set_visible(False)
          plotImgs.append(plotImg)
        else:
          plotImgs[x].set_data(f_glimpse_images[y, 0, x])
          plotImgs[x].autoscale()

      fillList = False

      ax = fig.add_subplot(144)
      ax.imshow(np.reshape(nextX[0],(28,28)), cmap=plt.get_cmap('gray'))
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
      loc = ((sampled_locs_fetched[0,y,:]+1)*14).astype(int)
      ax.add_patch(patches.Rectangle(np.flipud(loc)-np.array([2,2]),4,4,fill=False,linestyle='solid',color=colors[eq]))

      fig.canvas.draw()
      fig.subplots_adjust(hspace=0)  #No horizontal space between subplots
      fig.subplots_adjust(wspace=0)  #No vertical space between subplots
      if save_dir is not None:
        plt.savefig(save_dir+'canvas'+str(nextY[0])+str(10+y)+'.png')
      time.sleep(0.1)
      plt.pause(0.0001)
    #ax.remove()
    return