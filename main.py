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
from typing import List
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
import imutils
import cv2
import keras.backend as K
import tensorflow as tf
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from keras.models import Sequential
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, \
    Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
# from keras_contrib.layers.normalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from collections import OrderedDict
# Removes error when running Tensorflow on GPU
# for Tensorflow 2.2 and Python 3.6+
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession


# ----------------------------------------
# SETUP GPU
# ----------------------------------------
# # Testing and enabling GPU
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------
# =============== FUNCTIONS===============
# ----------------------------------------

def setup_options():
    # Setup various options for the network such as dimensions,
    # learning rates etc.

    # size of the latent space
    latent_dim = 5


# ----------------------------------------
def load_dataset():
    # Loading the data. There will be two types:
    # 1. Polarized PNG files (0, 45, 90, 135 degree)
    # 2. Generate estimated diffuse image
    # 3. The images are already resized in the read_image function

    # filepath = os.path.realpath(__file__)
    filepath = "/home/atif/Documents/Datasets/PolarizedImages/"
    dirname = os.path.dirname(filepath)
    # The source folder for the polarized images
    sourceFolder = os.path.join(dirname, "rawPolarized")
    # The desitnation folder for the generated images
    destinationFolder = os.path.join(dirname, "generatedImgs")

    polarization_labels = ['0', '45', '90', '135']

    OriginalImageStack, height, width, channels = read_images(sourceFolder, pattern="*_Itot.png")
    imgStack_0deg, height, width, channels = read_images(sourceFolder, pattern="*_0.png")
    imgStack_45deg, height, width, channels = read_images(sourceFolder, pattern="*_45.png")
    imgStack_90deg, height, width, channels = read_images(sourceFolder, pattern="*_90.png")
    imgStack_135deg, height, width, channels = read_images(sourceFolder, pattern="*_135.png")
    imgStack_masks, height, width, channels = read_images(sourceFolder, pattern="*_mask.png")

    print("\nNo of images in folder: {0}".format(len(imgStack_0deg)))
    # ESTIMATED DIFFUSE CALCULATION:
    # Ideally, the diffuse should only be approximated in areas of specular highlight, leaving the other areas untouched
    # However for the time, we can just plug in the whole image and take the minimum
    # The Shen2009 method or Dark Channel Prior can be used to mask out specular highlight areas for estimating diffuse
    #  >>>>> TO DO: Add Specular Highlight Detection method for estimating Diffuse (Shen or DCP etc)
    b = []
    g = []
    r = []
    estimated_diffuse_stack = []
    i = 0
    for orig, img0, img45, img90, img135 in zip(OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg,
                                                imgStack_135deg):
        # Note: Each img variable is a 3 channel image; so we can split it up in BGR
        blue, green, red = cv2.split(img0)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img45)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img90)
        b.append(blue)
        g.append(green)
        r.append(red)

        blue, green, red = cv2.split(img135)
        b.append(blue)
        g.append(green)
        r.append(red)

        b_min = np.amin(b, axis=0)
        g_min = np.amin(g, axis=0)
        r_min = np.amin(r, axis=0)

        merged = cv2.merge([b_min, g_min, r_min])
        i += 1

        # WRITE the image to a file if required. Can eb commented out if req
        cv2.imwrite('ResultA0' + str(i) + '_min' + '.png', merged)

        # Stack the estimated diffuse images for later use in loop
        estimated_diffuse_stack.append(merged)

        # DEBUG STACK DISPLAY
        Horizontal1 = np.hstack([orig, merged])
        # Debug display
        cv2.namedWindow("Loaded Image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("dice", 600,600)
        cv2.imshow("Loaded Image", Horizontal1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # i += 1

        # clear data before next loop; avoiding any data overwriting issues
        b.clear()
        g.clear()
        r.clear()
        merged.fill(0)  # clear the vars before calculating
    # Returns all the 4xpolarized images and their estimated diffuse images (Total 5 images)
    return OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg, imgStack_135deg, estimated_diffuse_stack


# ----------------------------------------
# Read all images in the dataset and return a np array
def read_images(path, pattern):
    image_stack = []
    # build path string, sort by name
    for img in sorted(glob.glob(path + "/" + pattern)):
        img = cv2.imread(img)
        # Resize image to improve performance
        resized_image = resize_images(img, rowsize=256, colsize=256)
        image_stack.append(resized_image)

    height, width, channels = image_stack[0].shape
    return image_stack, height, width, channels


# ----------------------------------------
def resize_images(img, rowsize, colsize):
    # The loaded images will be resized to lower res for faster training and eval. After POC, higher res can be used
    # rows, cols, ch = img.shape
    # Adding white balance to remove the green tint generating from the polarized images
    resized_image = white_balance(imutils.resize(img, width=colsize, height=rowsize))
    return resized_image


# ------------------------------------------
# White balance
#  Source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
# ------------------------------------------
def white_balance(input_image):
    result = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    whiteBalancedImage = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return whiteBalancedImage


# ----------------------------------------
def define_descriminator():
    # define
    return


# ----------------------------------------
def plot_model(generator, to_file, show_shapes, show_layer_names):
    # plot model
    return


# ----------------------------------------
def define_generator(latent_dim):
    # generator
    return


# ----------------------------------------
def define_gan(generator_model, discriminator_model):
    #
    ganModel = 0
    return ganModel


# ----------------------------------------
def train(generator, discriminator, gan_model, latent_dim):
    # Main training model
    return


# ----------------------------------------
# ================= MAIN =================
# ----------------------------------------

# To setup the various variables for the network
setup_options()
# Load the dataset with resized polar and estimated diffuse images
OriginalImageStack, img_0deg, img_45deg, img_90deg, img_135deg, estimated_diffuse = load_dataset()
print("[LOADED IMAGE] - Dataset size: ", len(img_0deg), "images")
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
