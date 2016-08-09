# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:15:43 2016

@author: rob
"""

import numpy as np
import matplotlib.pyplot as plt

def translate(im_batch, width,vis=False):
  """Randomly translates MNIST images in a [width x width]
  black image.
  Set vis=True to visualize the process
  Assumes square images"""
  assert len(im_batch.shape) == 2, 'Please provide a 2D np-array of size [batch_size, D]'
  if width == 0:
    return im_batch
  assert width>28, 'Please provide a width larger than 28'
  N,D2 = im_batch.shape
  D = np.sqrt(D2)
  assert D.is_integer(), 'Please provide a batch of square images'
  D = int(D)
  im_batch = np.reshape(im_batch, (N,D,D))

  if vis:
    plt.imshow(im_batch[0])

  IM = np.zeros((N,width,width))
  offsets = np.random.randint(0,width-D,size=(N,2))  #
  for n in range(N):
    IM[n,offsets[n,0]:offsets[n,0]+D,offsets[n,1]:offsets[n,1]+D] = im_batch[n]
  if vis:
    plt.imshow(IM[0])
  return np.reshape(IM,(N,width**2))