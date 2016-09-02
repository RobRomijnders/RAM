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
  def __init__(self,config, dataset = 'mnist'):
    assert dataset in ['cifar','mnist','tria'], 'For now, we only support cifar or mnist or tria dataset'
    self.dataset = dataset
    self.depth = config['depth'] # zooms
    min_radius = config['min_radius']
    max_radius = config['max_radius']
    self.sensorBW = config['sensorBandwidth'] # fixed resolution of sensor
    self.im_size = config['im_size']
    self.batch_size = config['batch_size']


    cell_size = 256
    cell_out_size = cell_size

    self.glimpses = config['glimpses']
    n_classes = config['n_classes']

    lr = config['lr']
    if dataset == 'mnist':
      self.channels = 1 # grayscale
      self.colormap = plt.get_cmap('gray')
    if dataset == 'tria':
      self.channels = 1
      self.colormap = plt.get_cmap('gray')
      self.im_size = 70
    elif dataset == 'cifar':
      self.channels = 3
      self.colormap = None
    ch_conv1 = self.depth*self.channels  #Number of channels feeding into CONV1
    ch_conv2 = 12  #HYPERPARAMETER: channels in CONV2
    g_size = self.sensorBW*self.sensorBW*ch_conv2
    loc_size = 4

    loc_sd_final = config['loc_sd_final']
    loc_sd_start = config['loc_sd_start']
    global_step = tf.Variable(0,trainable=False)
    loc_sd = tf.train.exponential_decay(loc_sd_start-loc_sd_final, global_step, 1000, 0.85)+ loc_sd_final


    mean_locs = []
    self.sampled_locs = [] # ~N(mean_locs[.], loc_sd)
    self.glimpse_images = [] # to show in window

    #Initialize parameters
    #naming according to
    # - W is a trainable weight-matrix
    # - b is a trainable bias-vector
    # l postfixes for LOCATION
    # g postfixes for GLIMPSE

    Wl1 = tf.get_variable("Wl1", shape=[2,loc_size],initializer=tf.contrib.layers.xavier_initializer())
    bl1 = tf.Variable(tf.constant(0.1, shape=[loc_size]),name='bl1')
    Wg1 = tf.get_variable("Wg1", shape=[3,3,ch_conv1,ch_conv2],initializer=tf.contrib.layers.xavier_initializer())
    bg1 = tf.Variable(tf.constant(0.1,shape=[ch_conv2]),name='bg1')
    Wg2 = tf.get_variable("Wg2", shape=[3,3,ch_conv2,ch_conv2],initializer=tf.contrib.layers.xavier_initializer())
    bg2 = tf.Variable(tf.constant(0.1,shape=[ch_conv2]),name='bg2')
    self.Wl_out = tf.get_variable("Wl_out", shape=[cell_out_size,2],initializer=tf.contrib.layers.xavier_initializer())
    bl_out = tf.Variable(tf.constant(0.1,shape=[2]),name='bl_out')


    def sensor_glimpse(img, normLoc):
      loc = ((normLoc + 1.0) / 2.0) * self.im_size
      loc = tf.cast(loc, tf.int32)

      #Random perturbations cannot exceed the image
      loc = tf.clip_by_value(loc, 0, self.im_size)


      zooms = []

      for k in xrange(self.batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        offset = max_radius

        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(one_img, offset, offset, \
            max_radius * 2 + self.im_size, max_radius * 2 + self.im_size)
        for i in xrange(self.depth):
          r = int(min_radius * (2 ** (i-1)))

          d_raw = 2 * r
          d = tf.constant(d_raw, shape=[1])

          loc_k = loc[k,:]
          d = tf.concat(0,[tf.tile(d, [2]),tf.constant(self.channels,shape=[1])])

          #Add broadcasting information for the channels
          adjusted_loc = offset + loc_k - r
          adjusted_loc = tf.concat(0,[adjusted_loc,tf.constant(0,shape=[1])])

          zoom = tf.slice(one_img, adjusted_loc, d, name='crop_image')
          # Resize_bilinear or resize_nearest_neighbor broadcasts across channel-dimenstion. Though it expects a rank-4 Tensor
          # User can choose between resize_bilinear and other resize operations. For RGB values, I find that nearest neighbor resizing
          # preserves the image better, because bilinear or cubic interpolation is not intuitive for RGB vectors
          zoom = tf.image.resize_nearest_neighbor(tf.expand_dims(zoom, 0), (self.sensorBW, self.sensorBW))
          zoom = tf.squeeze(zoom,[0])
          imgZooms.append(zoom)
        #Concatenate imgZooms across channel-dimension
        zooms.append(tf.concat(2,imgZooms))
      #Pack zooms across batch_size-dimension
      zooms = tf.pack(zooms)
      zooms = tf.stop_gradient(zooms)  #Stop the gradient here, so the classifier won't backprop into the location network
      self.glimpse_images.append(zooms)

      return zooms

    def get_glimpse(loc):
      loc = tf.stop_gradient(loc)
      glimpse_input = sensor_glimpse(img, loc)
      hg1 = tf.nn.relu(tf.nn.conv2d(glimpse_input, Wg1,[1,1,1,1],'SAME')+bg1)
      hg2 = tf.nn.relu(tf.nn.conv2d(hg1, Wg2,[1,1,1,1],'SAME')+bg2)
      hl = tf.nn.relu(tf.nn.xw_plus_b(loc, Wl1,bl1))
      g = tf.concat(1,[hl, tf.reshape(hg2,[self.batch_size,g_size])])
      return g

    def get_next_input(output, i):
      mean_loc = tf.tanh(tf.nn.xw_plus_b(output, self.Wl_out,bl_out))
      mean_locs.append(mean_loc)
      sample_loc = mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd)
      self.sampled_locs.append(sample_loc)
      return get_glimpse(sample_loc)

    def gaussian_pdf(mean, sample):
      Z = 1.0 / (loc_sd * tf.sqrt(2.0 * math.pi))
      a = -tf.reduce_sum(tf.square(sample-mean),2) / (2.0 * tf.square(loc_sd))
      return Z * tf.exp(a)
    if self.dataset  == 'mnist':
      self.image = tf.placeholder(tf.float32, shape=(self.batch_size, self.im_size * self.im_size), name="images")
      img = tf.reshape(self.image, (self.batch_size, self.im_size, self.im_size, self.channels))
    elif self.dataset == 'cifar':
      self.image = tf.placeholder(tf.float32, shape = (self.batch_size, 3*self.im_size*self.im_size), name = 'images')
      img = tf.transpose(tf.reshape(self.image, (self.batch_size, 3, self.im_size, self.im_size)),perm=[0,2,3,1])
    elif self.dataset == 'tria':
      self.image = tf.placeholder(tf.float32, shape=(self.batch_size, self.im_size, self.im_size), name="images")
      img = tf.reshape(self.image, (self.batch_size, self.im_size, self.im_size, self.channels))


    self.labels = tf.placeholder(tf.int64, shape=(self.batch_size), name="labels")
    self.keep_prob = tf.placeholder("float", name = 'Drop_out')



    initial_loc = tf.constant(0.0,dtype=tf.float32,shape=[self.batch_size,2])

    initial_glimpse = get_glimpse(initial_loc)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(cell_size,state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
    initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)

    inputs = [initial_glimpse]
    inputs.extend([0] * (self.glimpses - 1))

    outputs, _ = tf.nn.seq2seq.rnn_decoder(inputs, initial_state, lstm_cell, loop_function=get_next_input)
    get_next_input(outputs[-1], 0)


    # convert list of tensors to one big tensor
    self.sampled_locs = tf.pack(self.sampled_locs)
    self.sampled_locs = tf.transpose(self.sampled_locs, [1,0,2])
    mean_locs = tf.pack(mean_locs)
    mean_locs = tf.transpose(mean_locs, [1,0,2])
    # both mean_locs and sampled_locs now in [batch_size, glimpses,2]
    self.glimpse_images = tf.concat(0, self.glimpse_images)


    outputs = outputs[-1] # look at ONLY THE END of the sequence
    Wpred_out = tf.get_variable("Wpred_out", shape=[cell_out_size, n_classes],initializer=tf.contrib.layers.xavier_initializer())
    bpred_out = tf.Variable(tf.constant(0.1, shape=[n_classes]),name='bpred_out')

    a_y = tf.nn.xw_plus_b(outputs,Wpred_out,bpred_out)
    cost_sm = tf.nn.sparse_softmax_cross_entropy_with_logits(a_y, self.labels)
    self.labels_pred = tf.arg_max(a_y, 1)

    R = tf.cast(tf.equal(self.labels_pred, self.labels), tf.float32) # reward per example

    self.reward = tf.reduce_mean(R) # overall reward

    p_loc = gaussian_pdf(mean_locs, tf.stop_gradient(self.sampled_locs))

    R = tf.reshape(R, (self.batch_size, 1))
    J = tf.log(p_loc + 1e-9) * tf.stop_gradient(R)
    J = tf.reduce_sum(J, 1)
#    tvars = tf.trainable_variables()
#    self.grads = tf.gradients(J,tvars)
#    grads_zip = zip(self.grads,tvars)
#    for g,variable in grads_zip:
#      print(variable.name)
    J_comb = cost_sm - J
    self.debug = tf.gradients(cost_sm,Wg1)
    J_comb = tf.reduce_mean(J_comb, 0)
    self.cost = J_comb

    lrate = tf.train.exponential_decay(lr,global_step,10000,0.9,staircase=False)

    optimizer = tf.train.AdamOptimizer(lrate)
    self.train_op = optimizer.minimize(self.cost, global_step=global_step)


    tf.scalar_summary("reward", self.reward)
    tf.scalar_summary("cost", self.cost)

    self.summary_op = tf.merge_all_summaries()
    prefix = tf.expand_dims(initial_loc,1)
    self.locs = tf.concat(1,[prefix,self.sampled_locs])
    print('Finished computation graph')

  def draw_ram(self,f_glimpse_images_fetched,prediction_labels_fetched,sampled_locs_fetched,nextX,nextY,save_dir=None):
    colors = ['r','g']
    fig = plt.figure()
    plt.show()
    plotImgs = []
    if self.dataset == 'mnist':
      f_glimpse_images = np.reshape(f_glimpse_images_fetched, (self.glimpses + 1, self.batch_size, self.sensorBW, self.sensorBW, self.depth)) #steps, THEN batch
    elif self.dataset == 'cifar':
      f_glimpse_images = np.reshape(f_glimpse_images_fetched, (self.glimpses + 1, self.batch_size, self.sensorBW, self.sensorBW,self.depth*self.channels)) #steps, THEN batch
    elif self.dataset == 'tria':
      f_glimpse_images = np.reshape(f_glimpse_images_fetched, (self.glimpses + 1, self.batch_size, self.sensorBW, self.sensorBW,self.depth*self.channels)) #steps, THEN batch
    else:
      print('Wrong dataset used')

    fillList = False

    if len(plotImgs) == 0:
      fillList = True
    ind = np.random.choice(self.batch_size)
    # display first in mini-batch
    for y in xrange(self.glimpses):
      eq = int(prediction_labels_fetched[ind] ==  nextY[ind])
      for x in xrange(self.depth):
        plt.subplot(1, 4, x + 1)
#        print(fillList)
        if self.dataset == 'mnist':
          img_add = f_glimpse_images[y, ind,:,:, x]
        elif self.dataset == 'cifar':
          img_add = f_glimpse_images[y,ind,:,:,x*self.channels:(x+1)*self.channels]
        elif self.dataset == 'tria':
          img_add = f_glimpse_images[y, ind,:,:, x]
        if fillList:
          plotImg = plt.imshow(img_add, interpolation="nearest", shape=(self.im_size,self.im_size), cmap = self.colormap)
          plotImg.autoscale()
          plotImg.axes.get_xaxis().set_visible(False)
          plotImg.axes.get_yaxis().set_visible(False)
          plotImgs.append(plotImg)
        else:
          plotImgs[x].set_data(img_add)
          plotImgs[x].autoscale()

      fillList = False
      ax = fig.add_subplot(144)
      if self.dataset == 'mnist':
        im_show = np.reshape(nextX[ind],(self.im_size,self.im_size))
      elif self.dataset == 'cifar':
        im_show = np.transpose(np.reshape(nextX[ind],(3,self.im_size,self.im_size)),(1,2,0))
      elif self.dataset == 'tria':
        im_show = nextX[ind]
      ax.imshow(im_show, cmap = self.colormap)
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
      loc = ((sampled_locs_fetched[ind,y,:]+1)*self.im_size/2).astype(int)
      ax.add_patch(patches.Rectangle(np.flipud(loc)-np.array([2,2]),4,4,fill=False,linestyle='solid',color=colors[eq]))

      fig.canvas.draw()
      fig.subplots_adjust(hspace=0)  #No horizontal space between subplots
      fig.subplots_adjust(wspace=0)  #No vertical space between subplots
      if save_dir is not None:
        plt.savefig(save_dir+'canvas'+str(nextY[ind])+str(10+y)+'.png')
      time.sleep(0.1)
      plt.pause(0.0001)
    #ax.remove()
    return

#    sensorArea = self.sensorBW**2
#    BW = self.depth * self.sensorBW * self.sensorBW * channels  #Bandwidth
#    hg_size = 128
#    hl_size = 128