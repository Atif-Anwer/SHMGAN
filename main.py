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
import argparse
import glob
import os
# import cv2
# import numpy as np
# from matplotlib import pyplot
import h5py
import tensorflow as tf
# noinspection PyUnresolvedReferences
from keras.layers import Concatenate, Lambda, Reshape, Add, LeakyReLU
from keras.layers import Conv2D, Input, ReLU, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as KERAS_backend
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# from keras_contrib.layers.normalization import InputSpec
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# Removes error when running Tensorflow on GPU
# for Tensorflow 2.2 and Python 3.6+
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from keras.utils.vis_utils import plot_model
from functools import partial
from utils import *
import random


# ----------------------------------------
# =============== FUNCTIONS===============
# ----------------------------------------

def parse_args():
    # Setup various options for the network such as dimensions,
    # learning rates etc.

    desc = "Specular Highlight Mitigation from Polarimetric Images - SHMGAN"
    parser = argparse.ArgumentParser( description = desc )

    # Model configuration.
    parser.add_argument( '--num_iteration', type = int, default = 200000, help = 'number of total iterations for training D' )
    parser.add_argument( '--est_diffuse', type = bool, default = True,
                         help = '(TRUE) Estimate diffuse image from images or (FALSE) load from hdf5 file' )
    parser.add_argument( '--image_size', type = int, default = 128, help = 'image resize resolution' )
    parser.add_argument( '--5', type = int, default = 5, help = 'dimension of polarimetric domain images )' )
    parser.add_argument( '--batch_size', type = int, default = 4, help = 'mini-batch size' )
    parser.add_argument( '--g_lr', type = float, default = 0.0001, help = 'learning rate for G' )
    parser.add_argument( '--d_lr', type = float, default = 0.0001, help = 'learning rate for D' )
    parser.add_argument( '--beta1', type = float, default = 0.5, help = 'beta1 for Adam optimizer' )
    parser.add_argument( '--beta2', type = float, default = 0.999, help = 'beta2 for Adam optimizer' )
    parser.add_argument( '--selected_attrs', '--list', nargs = '+', help = 'selected attributes for the CelebA dataset',
                         default = ['0deg', '45deg', '90deg', '135deg', 'est_diffuse'] )
    parser.add_argument( '--num_iteration_decay', type = int, default = 100000, help = 'number of iterations for decaying lr' )
    parser.add_argument( '--n_critic', type = int, default = 5, help = 'number of D updates per each G update' )
    parser.add_argument( '--lambda_cls', type = float, default = 1, help = 'weight for domain classification loss' )
    parser.add_argument( '--lambda_rec', type = float, default = 10, help = 'weight for reconstruction loss' )
    parser.add_argument( '--lambda_gp', type = float, default = 10, help = 'weight for gradient penalty' )

    # Directories.
    # parser.add_argument( "-p", "--path", default = "./home/atif/Documents/Datasets/SHMGAN_dataset/", help = "Path to polarimetric image" )
    parser.add_argument( "-p", "--path", default = "/home/atif/Documents/Datasets/KAUST/", help = "Path to polarimetric image" )
    parser.add_argument( '--model_save_dir', type = str, default = 'models' )
    parser.add_argument( '--sample_dir', type = str, default = 'samples' )
    parser.add_argument( '--result_dir', type = str, default = 'results' )

    # Step size.
    parser.add_argument( '--log_step', type = int, default = 10 )
    parser.add_argument( '--sample_step', type = int, default = 1000 )
    parser.add_argument( '--model_save_step', type = int, default = 10000 )
    parser.add_argument( '--lr_update_step', type = int, default = 1000 )

    # Miscellaneous.
    parser.add_argument( '--mode', type = str, default = 'train', choices = ['train', 'test', 'custom'] )

    print( '\n [ => ] Passing all input arguments...' )
    return parser.parse_args()


# ----------------------------------------
def load_dataset( args ):
    # Loading the data. There will be two types:
    # 1. Polarized PNG files (0, 45, 90, 135 degree)
    # 2. Generate estimated diffuse image
    # 3. The images are already resized in the read_image function

    filepath = args.path
    dirname = os.path.dirname( filepath )
    # The source folder for the polarized images
    sourceFolder = os.path.join( dirname, "PolarImages" )
    # The destination folder for the generated images
    destinationFolder = os.path.join( dirname, "GeneratedImgs" )

    polarization_labels = ['0', '45', '90', '135']
    image_size = args.image_size
    filenames_Itot, OriginalImageStack, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_Itot.png" )
    filenames_0deg, imgStack_0deg, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_0.png" )
    filenames_45deg, imgStack_45deg, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_45.png" )
    filenames_90deg, imgStack_90deg, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_90.png" )
    filenames_135deg, imgStack_135deg, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_135.png" )
    filenames_masks, imgStack_masks, height, width, channels = read_images( sourceFolder, image_size, pattern = "*_mask.png" )

    print( "\n No of images in folder: {0}".format( len( imgStack_0deg ) ) )
    # ESTIMATED DIFFUSE CALCULATION:
    # Ideally, the diffuse should only be approximated in areas of specular highlight, leaving the other areas untouched
    # However for the time, we can just plug in the whole image and take the minimum
    # The Shen2009 method or Dark Channel Prior can be used to mask out specular highlight areas for estimating diffuse
    #  >>>>> TO DO: Add Specular Highlight Detection method for estimating Diffuse (Shen or DCP etc)
    b = []
    g = []
    r = []
    estimated_diffuse_stack = []
    filenames_est_diffuse = []
    i = 0

    if args.est_diffuse:  # If argument is 'True' calculate the estimated diffuse images
        for orig, img0, img45, img90, img135 in zip( OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg,
                                                     imgStack_135deg ):
            # Note: Each img variable is a 3 channel image; so we can split it up in BGR
            blue, green, red = cv2.split( img0 )
            b.append( blue )
            g.append( green )
            r.append( red )

            blue, green, red = cv2.split( img45 )
            b.append( blue )
            g.append( green )
            r.append( red )

            blue, green, red = cv2.split( img90 )
            b.append( blue )
            g.append( green )
            r.append( red )

            blue, green, red = cv2.split( img135 )
            b.append( blue )
            g.append( green )
            r.append( red )

            b_min = np.amin( b, axis = 0 )
            g_min = np.amin( g, axis = 0 )
            r_min = np.amin( r, axis = 0 )

            merged = cv2.merge( [b_min, g_min, r_min] )
            i += 1

            # WRITE the image to a file if required. Can eb commented out if req
            # TODO: Save Estimated Diffuse Images in separate folder
            name = 'ResultA0' + str( i ) + '_min' + '.png'
            # cv2.imwrite(name, merged)
            filenames_est_diffuse.append( name )

            # Stack the estimated diffuse images for later use in loop
            estimated_diffuse_stack.append( merged )
    #
    # # DEBUG STACK DISPLAY
    # Horizontal1 = np.hstack([orig, merged])
    # # Debug display
    # cv2.namedWindow("Loaded Image", cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("dice", 600,600)
    # cv2.imshow("Loaded Image", Horizontal1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # i += 1
    #
    # # clear data before next loop; avoiding any data overwriting issues
    # b.clear()
    # g.clear()
    # r.clear()
    # merged.fill(0)  # clear the vars before calculating
    else:  # If argument is 'false' then load from hdf5 file
        save_dataset_hdf5( OriginalImageStack )

    print( '\n [ => ] Loading dataset, calculating estimated diffuse and returning filenames ...' )
    # Returns all the 4xpolarized images and their estimated diffuse images (Total 5 images)
    return filenames_Itot, filenames_0deg, filenames_45deg, filenames_90deg, filenames_135deg, filenames_est_diffuse


# ----------------------------------------
def save_dataset_hdf5( image_stack ):
    save_path = './estimated_diffuse_images.hdf5'
    hf = h5py.File( save_path, 'a' )  # open a hdf5 file

    dset = hf.create_dataset( 'default', data = image_stack, compression = "gzip", compression_opts = 9 )
    hf.close()  # close the hdf5 file
    print( '\n [ => ] Dataset Saved. hdf5 file size: %d bytes' % os.path.getsize( save_path ) )


# ----------------------------------------
# Read all images in the dataset and return a np array
def read_images( path, new_size, pattern ):
    image_stack = []
    filenames = []
    # build path string, sort by name
    for img_path in sorted( glob.glob( path + "/" + pattern ) ):
        img = cv2.imread( img_path )
        # Resize image to improve performance
        resized_image = resize_images( img, rowsize = new_size, colsize = new_size )
        image_stack.append( resized_image )
        head, tail = os.path.split( img_path )
        # filenames.append( tail )
        filenames.append( img_path )

    height, width, channels = image_stack[0].shape
    return filenames, image_stack, height, width, channels


# ----------------------------------------
def define_discriminator( options ):
    # Discriminator network with PatchGAN

    image_size = options.image_size
    inp_img = Input( shape = (image_size, image_size, 3) )
    x = ZeroPadding2D( padding = 1 )( inp_img )
    x = Conv2D( filters = 64, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False )( x )
    x = LeakyReLU( 0.01 )( x )

    # 6 conv layers
    N = 64
    for i in range( 1, 6 ):
        x = ZeroPadding2D( padding = 1 )( x )
        x = Conv2D( filters = N * 2, kernel_size = 4, strides = 2, padding = 'valid' )( x )
        x = LeakyReLU( 0.01 )( x )
        N = N * 2

    kernel_size = int( image_size / np.power( 2, 6 ) )

    out_src = ZeroPadding2D( padding = 1 )( x )
    out_src = Conv2D( filters = 1, kernel_size = 3, strides = 1, padding = 'valid', use_bias = False )( out_src )

    out_cls = Conv2D( filters = 5, kernel_size = kernel_size, strides = 1, padding = 'valid', use_bias = False )( x )
    out_cls = Reshape( (5,) )( out_cls )

    print( '\n [ => ] Building Discriminator ...' )
    return Model( inp_img, [out_src, out_cls] )


# ----------------------------------------
def define_generator( options ):
    # generator from StarGAN
    """Generator network."""
    # Input tensors
    image_size = options.image_size
    inp_c = Input( shape = 5 )
    inp_img = Input( shape = (image_size, image_size, 3) )

    # Replicate spatially and concatenate domain information
    c = Lambda( lambda x: KERAS_backend.repeat( x, image_size ** 2 ) )( inp_c )
    c = Reshape( (image_size, image_size, 5) )( c )
    g = Concatenate()( [inp_img, c] )

    # First Conv2D
    g = Conv2D( filters = 64, kernel_size = 7, strides = 1, padding = 'same', use_bias = False )( g )
    g = InstanceNormalization( axis = -1 )( g )
    g = ReLU()( g )

    # Down-sampling layers
    N = 64
    for i in range( 2 ):
        g = ZeroPadding2D( padding = 1 )( g )
        g = Conv2D( filters = N*2, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False )( g )
        g = InstanceNormalization( axis = -1 )( g )
        g = ReLU()( g )
        N *= 2  # double the filters for next layer

    # Bottleneck layers.
    # 6 layers of 256, K 3x3, S1, P1, ReLU
    for i in range( 6 ):
        g = ResidualBlock( g, N )

    # Up-sampling layers
    # 3 upsampling layers, with 256, 128, 64 filters respectively
    for i in range( 2 ):
        g = UpSampling2D( size = 2 )( g )
        g = Conv2D( filters = N // 2, kernel_size = 4, strides = 1, padding = 'same', use_bias = False )( g )
        g = InstanceNormalization( axis = -1 )( g )
        g = ReLU()( g )
        N //= 2

    # Last Conv2D
    g = ZeroPadding2D( padding = 3 )( g )
    out = Conv2D( filters = 3, kernel_size = 7, strides = 1, padding = 'valid', activation = 'tanh', use_bias = False )( g )

    print( '\n [ => ] Building Generator ...' )
    return Model( inputs = [inp_img, inp_c], outputs = out )


# ----------------------------------------
def ResidualBlock( inp, dim_out ):
    """Residual Block with instance normalization."""
    x = ZeroPadding2D( padding = 1 )( inp )
    x = Conv2D( filters = dim_out, kernel_size = 3, strides = 1, padding = 'valid', use_bias = False )( x )
    x = InstanceNormalization( axis = -1 )( x )
    x = ReLU()( x )
    x = ZeroPadding2D( padding = 1 )( x )
    x = Conv2D( filters = dim_out, kernel_size = 3, strides = 1, padding = 'valid', use_bias = False )( x )
    x = InstanceNormalization( axis = -1 )( x )
    return Add()( [inp, x] )


# ----------------------------------------
def gradient_penalty_loss( self, y_true, y_pred, averaged_samples ):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = KERAS_backend.gradients( y_pred, averaged_samples )[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = KERAS_backend.square( gradients )
    #   ... summing over the rows ...
    gradients_sqr_sum = KERAS_backend.sum( gradients_sqr, axis = np.arange( 1, len( gradients_sqr.shape ) ) )
    #   ... and sqrt
    gradient_l2_norm = KERAS_backend.sqrt( gradients_sqr_sum )
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = KERAS_backend.square( 1 - gradient_l2_norm )
    # return the mean as loss over all the batch samples
    return KERAS_backend.mean( gradient_penalty )


# ----------------------------------------
def RandomWeightedAverage( bs, inputs ):
    """Provides a (random) weighted average between real and generated image samples"""
    alpha = KERAS_backend.random_uniform( (bs, 1, 1, 1) )
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# -------LOSSES---------------------------
# ----------------------------------------
def wasserstein_loss( self, Y_true, Y_pred ):
    return KERAS_backend.mean( Y_true * Y_pred )


def reconstruction_loss( self, Y_true, Y_pred ):
    return KERAS_backend.mean( KERAS_backend.abs( Y_true - Y_pred ) )


def classification_loss( self, Y_true, Y_pred ):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred)) # orig
    # return tf.reduce_mean( tf.math.squared_difference( Y_pred, Y_true ) )
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits_v2(labels=Y_true, logits=Y_pred))
    # return tf.reduce_mean(tf.losses.categorical_crossentropy(Y_true, Y_pred, from_logits=True))


# ----------------------------------------
def train( args, train_D, train_G, train_dataset, train_dataset_label, D, G ):
    # Main training model
    data_iter = get_loader( train_dataset, train_dataset_label, image_size = args.image_size, batch_size = args.batch_size, mode = args.mode )

    valid = np.ones( (args.batch_size, 2, 2, 1) )
    fake = np.ones( (args.batch_size, 2, 2, 1) )
    dummy = np.zeros( (args.batch_size, 2, 2, 1) )  # Dummy gt for gradient penalty
    for epoch in range( args.num_iteration ):
        imgs, orig_labels, target_labels, fix_labels, _ = next( data_iter )

        # Setting learning rate (linear decay)
        if epoch > (args.num_iteration - args.num_iteration_decay):
            KERAS_backend.set_value( train_D.optimizer.lr,
                                     args.d_lr * (args.num_iteration - epoch) / (args.num_iteration - args.num_iteration_decay) )
            KERAS_backend.set_value( train_G.optimizer.lr,
                                     args.g_lr * (args.num_iteration - epoch) / (args.num_iteration - args.num_iteration_decay) )

        # Training Discriminators
        D_loss = train_D.train_on_batch( x = [imgs, target_labels], y = [valid, orig_labels, fake, dummy] )

        # Training Generators
        if (epoch + 1) % args.n_critic == 0:
            G_loss = train_G.train_on_batch( x = [imgs, orig_labels, target_labels], y = [valid, target_labels, imgs] )

        if (epoch + 1) % args.log_step == 0:
            print( f"Iteration: [{epoch + 1}/{args.num_iteration}]" )
            print(
                    f"\tD/loss_real = [{D_loss[1]:.4f}], D/loss_fake = [{D_loss[3]:.4f}], D/loss_cls =  [{D_loss[2]:.4f}], D/loss_gp = [{D_loss[4]:.4f}]" )
            print( f"\tG/loss_fake = [{G_loss[1]:.4f}], G/loss_rec = [{G_loss[3]:.4f}], G/loss_cls = [{G_loss[2]:.4f}]" )

        if (epoch + 1) % args.model_save_step == 0:
            G.save_weights( os.path.join( args.model_save_dir, 'G_weights.hdf5' ) )
            D.save_weights( os.path.join( args.model_save_dir, 'D_weights.hdf5' ) )
            train_D.save_weights( os.path.join( args.model_save_dir, 'train_D_weights.hdf5' ) )
            train_G.save_weights( os.path.join( args.model_save_dir, 'train_G_weights.hdf5' ) )


# ----------------------------------------
def test( generator, discriminator, gan_model, latent_dim ):
    # Main training model
    return


def check_gpu():
    # ----------------------------------------
    # SETUP GPU
    # ----------------------------------------
    # # Testing and enabling GPU
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print( tf.test.is_built_with_cuda() )
    print( "[ => ] Num GPUs Available: ", len( tf.config.list_physical_devices( 'GPU' ) ) )
    tf.config.list_physical_devices( 'GPU' )

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession( config = config )


# ----------------------------------------
# ================= MAIN =================
# ----------------------------------------
def main():
    # To setup the various variables for the network
    args = parse_args()

    # check gpu availability
    check_gpu()

    # -------------- LOAD DATASET -------------------------------
    # -----------------------------------------------------------
    # Load the dataset with resized polar and estimated diffuse images
    filenames_Itot, filenames_0deg, filenames_45deg, filenames_90deg, filenames_135deg, filenames_est_diffuse = load_dataset( args )
    print( "[LOADED IMAGES] - Dataset size: ", len( filenames_0deg ), "images" )

    # ---------------BUILD MODEL---------------------------------
    # -----------------------------------------------------------
    image_size = args.image_size
    # ----------------------------------------
    # create the generator
    G = define_generator( args )
    # summarize the model
    G.summary()
    # ----------------------------------------
    # create discriminator
    D = define_discriminator( args )
    # summarize the model
    D.summary()

    # ----------------------------------------
    # plot the model
    plot_model( D, to_file = 'discriminator_plot.png', show_shapes = True, show_layer_names = True )
    plot_model( G, to_file = 'generator_plot.png', show_shapes = True, show_layer_names = True )

    # THE MODEL SHOULD BE CORRECT BASED ON THE ABOVE GENERATED MODEL FILES
    # THIS WILL BE A GOOD REVIEW BEFORE GOING AHEAD WITH THE TRAINING
    # ----------------------------------------
    G.trainable = False
    # create the gan
    # ----------------------------------------
    # Compute output with real images.
    x_real = Input( shape = (image_size, image_size, 3) )
    out_src_real, out_cls_real = D( x_real )

    # Compute output with fake images.
    label_trg = Input( shape = (5,) )
    x_fake = G( [x_real, label_trg] )
    out_src_fake, out_cls_fake = D( x_fake )

    # Compute output for gradient penalty
    x_hat = RandomWeightedAverage( args.batch_size, [x_real, x_fake] )
    out_src, _ = D( x_hat )

    # Use Python partial to provide loss function with additional 'averaged_samples' argument
    partial_gp_loss = partial( gradient_penalty_loss, averaged_samples = x_hat )
    partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

    # Define training model D
    train_D = Model( [x_real, label_trg], [out_src_real, out_cls_real, out_src_fake, out_src] )

    # Setup loss for train_D
    train_D.compile( loss = [wasserstein_loss, classification_loss, wasserstein_loss, partial_gp_loss],
                     optimizer = Adam( learning_rate = args.d_lr, beta_1 = args.beta1, beta_2 = args.beta2 ),
                     loss_weights = [1, args.lambda_cls, 1, args.lambda_gp] )

    # Update G and not update D
    G.trainable = True
    D.trainable = False

    # All inputs
    real_x = Input( shape = (image_size, image_size, 3) )
    org_label = Input( shape = (5,) )
    trg_label = Input( shape = (5,) )

    # Compute output of fake image
    fake_x = G( [real_x, trg_label] )
    fake_out_src, fake_out_cls = D( fake_x )

    # Target-to-original domain.
    x_reconst = G( [fake_x, org_label] )

    # Define traning model G
    train_G = Model( [real_x, org_label, trg_label], [fake_out_src, fake_out_cls, x_reconst] )

    # Setup loss for train_G
    train_G.compile( loss = [wasserstein_loss, classification_loss, reconstruction_loss],
                     optimizer = Adam( learning_rate = args.g_lr, beta_1 = args.beta1, beta_2 = args.beta2 ),
                     loss_weights = [1, args.lambda_cls, args.lambda_rec] )

    # ----------------TRAIN AND TEST THE MODEL-------------------
    # -----------------------------------------------------------
    """ Input Image"""
    # TODO : Update preprocess with loading filenames and returning data for training
    test_dataset, test_dataset_label, train_dataset, train_dataset_label = preprocess(
        filenames_Itot, filenames_0deg, filenames_45deg,
        filenames_90deg, filenames_135deg, filenames_est_diffuse )

    # TODO : Fully Working StarGAN
    # TODO : Git-branch for collagan and shm codes
    # TODO : Update to include COLLAGAN Changes
    # TODO : Update to include SHMGAN losses
    if args.mode == 'train':
        train( args, train_D, train_G, train_dataset, train_dataset_label, D, G )
        print( " [*] Training finished!" )

    if args.mode == 'test':
        test()
        print( " [*] Test finished!" )


# ----------------------------------
if __name__ == '__main__':
    main()