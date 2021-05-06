"""
## SHMGAN -  Removal of Specular Highlights by a GAN Network
Copyright: Atif Anwer

Parameters
----------


## Requirements:
-------


## Dataset
--------

## Uses Packages:
Python 3.8
CUDA 11.3
cuDnn 8.0
"""

from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
import matplotlib as plt
import numpy as np
import random
import datetime
import time
import json
import math
import sys
import os
import glob

import cv2
import keras.backend as K
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
# from keras_contrib.layers.normalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

from collections import OrderedDict


# ----------------------------------------
# SETUP GPU
# ----------------------------------------
# # Testing and enabling GPU
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

# Removes error when running Tensorflow on GPU
# for Tensorflow 2.2 and Python 3.6+
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ----------------------------------------
# =============== FUNCTIONS===============
# ----------------------------------------

def setup_options():
    # Setup various options for the network such as dimensions,
    # learning rates etc.

    # size of the latent space
    latent_dim = 5

def load_dataset():
    # Loading the data. There will be two types:
    # 1. Polarized PNG files (0, 45, 90, 135 degree)
    # 2. Generate estimated diffuse image
    # After loading the dataset, we can resize the images to smaller resolution.
    # Starting small resolution for a faster training and then extend to bigger size later

    # filepath = os.path.realpath(__file__)
    filepath = "/home/atif/Documents/Datasets/PolarizedImages/"
    dirname = os.path.dirname(filepath)
    # The source folder for the polarized images
    sourceFolder = os.path.join(dirname, "rawPolarized")
    # The desitnation folder for the generated images
    destinationFolder = os.path.join(dirname, "generatedImgs")

    polarization_labels = ['0', '45', '90', '135']


    imgStack_0deg = read_images(sourceFolder, pattern="*_0.png")
    imgStack_45deg = read_images(sourceFolder, pattern="*_45.png")
    imgStack_90deg = read_images(sourceFolder, pattern="*_90.png")
    imgStack_135deg = read_images(sourceFolder, pattern="*_135.png")
    imgStack_masks = read_images(sourceFolder, pattern="*_mask.png")

    print("\nNo of images in folder: {0}".format(len(imgStack_0deg)))
    ## ESTIMATED DIFFUSE CALCULATION:
    # Ideally, the diffuse should only be approximated in areas of specular highlight, leaving the other areas untouched.
    # However for the time, we can just plug in the whole image and take the minimum
    # The Shen2009 method or Dark Channel Prior can be used to mask out specular highlight areas for estimating diffuse
    #  >>>>> TO DO: Add Specular Highlight Detection method for estimating Diffuse (Shen or DCP etc)
    for img in imgStack_0deg:
        b, g, r = cv2.split(img)
    b, g, r = cv2.split(img)

    return img_0deg, img_45deg, img_90deg, img_135deg, estimated_diffuse

def read_images(path, pattern):
    image_stack = []
    # build path string, sort by name
    for img in sorted(glob.glob(path + "/" + pattern)):
        image_stack.append(cv2.imread(img))
    return image_stack

def resize_images(rowsize, colsize):
    # The loaded images will
    return

def define_descriminator():
    # define
    return

def plot_model(generator, to_file, show_shapes, show_layer_names):
    # plot model
    return

def define_generator(latent_dim):
    # generator
    return

def define_gan(generator_model, discriminator_model):
    #
    ganModel = 0
    return ganModel

def train(generator, discriminator, gan_model, latent_dim):
    # Main training model
    return



# ----------------------------------------
# ================= MAIN =================
# ----------------------------------------

# To setup the various variables for the network
setup_options()
# Load the dataset with resized polar and estimated diffuse images
img_0deg, img_45deg, img_90deg, img_135deg, estimated_diffuse = load_dataset()
latent_dim = 5
# ----------------------------------------
# create discriminator
discriminator = define_descriminator()
# summarize the model
discriminator.summary()
# plot the model
plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
# ----------------------------------------
# create the generator
generator = define_generator(latent_dim)
# summarize the model
generator.summary()
# plot the model
plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)
# ----------------------------------------
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)
pyplot.savefig('Final_result.png')