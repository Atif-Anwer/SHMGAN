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
import cv2
import h5py
import imutils
import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from keras.layers import Concatenate, Lambda, Reshape, _Merge, Add, LeakyReLU
from keras.layers import Conv2D, Input, ReLU, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# from keras_contrib.layers.normalization import InputSpec
# from keras.utils import plot_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# Removes error when running Tensorflow on GPU
# for Tensorflow 2.2 and Python 3.6+
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from functools import partial
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
    parser.add_argument( "-p", "--path", default = "./home/atif/Documents/Datasets/PolarizedImages/", help = "Path to polarimetric image" )
    parser.add_argument( '--est_diffuse', type = bool, default = True,
                         help = '(TRUE) Estimate diffuse image from images or (FALSE) load from hdf5 file' )
    parser.add_argument( '--image_resize', type = int, default = 128, help = 'image resize resolution' )
    parser.add_argument( '--5', type = int, default = 5, help = 'dimension of polarimetric domain images )' )
    parser.add_argument( '--batch_size', type = int, default = 4, help = 'mini-batch size' )
    parser.add_argument( '--g_lr', type = float, default = 0.0001, help = 'learning rate for G' )
    parser.add_argument( '--d_lr', type = float, default = 0.0001, help = 'learning rate for D' )
    parser.add_argument( '--beta1', type = float, default = 0.5, help = 'beta1 for Adam optimizer' )
    parser.add_argument( '--beta2', type = float, default = 0.999, help = 'beta2 for Adam optimizer' )
    parser.add_argument( '--data_dir', type = str, default = 'data/celeba' )
    parser.add_argument( '--selected_attrs', '--list', nargs = '+', help = 'selected attributes for the CelebA dataset',
                         default = ['0deg', '45deg', '90deg', '135deg', 'est_diffuse'] )

    return parser.parse_args()


# ----------------------------------------
def load_dataset( args ):
    # Loading the data. There will be two types:
    # 1. Polarized PNG files (0, 45, 90, 135 degree)
    # 2. Generate estimated diffuse image
    # 3. The images are already resized in the read_image function

    # filepath = os.path.realpath(__file__)
    filepath = args.path
    dirname = os.path.dirname( filepath )
    # The source folder for the polarized images
    sourceFolder = os.path.join( dirname, "rawPolarized" )
    # The destination folder for the generated images
    destinationFolder = os.path.join( dirname, "generatedImgs" )

    polarization_labels = ['0', '45', '90', '135']
    image_resize = args.image_size
    OriginalImageStack, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_Itot.png" )
    imgStack_0deg, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_0.png" )
    imgStack_45deg, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_45.png" )
    imgStack_90deg, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_90.png" )
    imgStack_135deg, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_135.png" )
    imgStack_masks, height, width, channels = read_images( sourceFolder, image_resize, pattern = "*_mask.png" )

    print( "\nNo of images in folder: {0}".format( len( imgStack_0deg ) ) )
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
            # cv2.imwrite('ResultA0' + str(i) + '_min' + '.png', merged)

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

    # Returns all the 4xpolarized images and their estimated diffuse images (Total 5 images)
    return OriginalImageStack, imgStack_0deg, imgStack_45deg, imgStack_90deg, imgStack_135deg, estimated_diffuse_stack


# ----------------------------------------
def save_dataset_hdf5( image_stack ):
    save_path = './estimated_diffuse_images.hdf5'
    hf = h5py.File( save_path, 'a' )  # open a hdf5 file

    dset = hf.create_dataset( 'default', data = image_stack, compression = "gzip", compression_opts = 9 )
    hf.close()  # close the hdf5 file
    print( 'hdf5 file size: %d bytes' % os.path.getsize( save_path ) )


# ----------------------------------------
# Read all images in the dataset and return a np array
def read_images( path, new_size, pattern ):
    image_stack = []
    # build path string, sort by name
    for img in sorted( glob.glob( path + "/" + pattern ) ):
        img = cv2.imread( img )
        # Resize image to improve performance
        resized_image = resize_images( img, rowsize = new_size, colsize = new_size )
        image_stack.append( resized_image )

    height, width, channels = image_stack[0].shape
    return image_stack, height, width, channels


# ----------------------------------------
def resize_images( img, rowsize, colsize ):
    # The loaded images will be resized to lower res for faster training and eval. After POC, higher res can be used
    # rows, cols, ch = img.shape
    # Adding white balance to remove the green tint generating from the polarized images
    resized_image = white_balance( imutils.resize( img, width = colsize, height = rowsize ) )
    return resized_image


# ------------------------------------------
# White balance
#  Source: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
# ------------------------------------------
def white_balance( input_image ):
    result = cv2.cvtColor( input_image, cv2.COLOR_RGB2LAB )
    avg_a = np.average( result[:, :, 1] )
    avg_b = np.average( result[:, :, 2] )
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    whiteBalancedImage = cv2.cvtColor( result, cv2.COLOR_LAB2RGB )
    return whiteBalancedImage


# ----------------------------------------
def define_descriminator( options ):
    init = tf.keras.RandomNormal( stddev = 0.02 )  # weight initialization
    # Discriminator network with PatchGAN

    image_size = options.image_resize
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

    return Model( inp_img, [out_src, out_cls] )


# ----------------------------------------
def define_generator( options ):
    # generator from StarGAN
    """Generator network."""
    # Input tensors
    image_size = options.image_resize
    inp_c = Input( shape = 5 )
    inp_img = Input( shape = (image_size, image_size, 3) )

    # Replicate spatially and concatenate domain information
    c = Lambda( lambda x: K.repeat( x, image_size ** 2 ) )( inp_c )
    c = Reshape( (image_size, image_size, 5) )( c )
    g = Concatenate()( [inp_img, c] )

    # First Conv2D
    g = Conv2D( g, filters = 64, kernel_size = 7, strides = 1, padding = 'same', use_bias = False )
    g = InstanceNormalization( axis = -1 )( g )
    g = ReLU()( g )

    # Down-sampling layers
    # 3 downsampling layers, with 64, 128, 256 filters respectively
    N = 64
    for i in range( 2 ):
        g = ZeroPadding2D( padding = 1 )( g )
        g = Conv2D( filters = N, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False )( g )
        g = InstanceNormalization( axis = -1 )( g )
        g = ReLU()( g )
        N *= 2  # double the filters for next layer

    # Bottleneck layers.
    # 6 layers of 256, K 3x3, S1, P1, ReLU
    for i in range( 6 ):
        g = ResidualBlock( g, 256 )

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
def define_gan( generator_model, discriminator_model ):
    #
    ganModel = 0
    return ganModel


# ----------------------------------------
def gradient_penalty_loss( self, y_true, y_pred, averaged_samples ):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients( y_pred, averaged_samples )[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square( gradients )
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum( gradients_sqr, axis = np.arange( 1, len( gradients_sqr.shape ) ) )
    #   ... and sqrt
    gradient_l2_norm = K.sqrt( gradients_sqr_sum )
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square( 1 - gradient_l2_norm )
    # return the mean as loss over all the batch samples
    return K.mean( gradient_penalty )


# ----------------------------------------
def RandomWeightedAverage():
    """Provides a (random) weighted average between real and generated image samples"""
    define_batch_size( self, bs )
    _merge_function(self, inputs )
    alpha = K.random_uniform( (bs, 1, 1, 1) )
    return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def wasserstein_loss( self, Y_true, Y_pred ):
    return K.mean( Y_true * Y_pred )


def reconstruction_loss( self, Y_true, Y_pred ):
    return K.mean( K.abs( Y_true - Y_pred ) )


def classification_loss( self, Y_true, Y_pred ):
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred)) # orig
    return tf.reduce_mean( tf.math.squared_difference( Y_pred, Y_true ) )
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits_v2(labels=Y_true, logits=Y_pred))
    # return tf.reduce_mean(tf.losses.categorical_crossentropy(Y_true, Y_pred, from_logits=True))


def ImageData( self, data_dir, selected_attrs ):
    self.selected_attrs = selected_attrs

    self.data_path = os.path.join( data_dir, 'images' )
    self.lines = open( os.path.join( data_dir, 'list_attr_celeba.txt' ), 'r' ).readlines()

    self.train_dataset = []
    self.train_dataset_label = []
    self.train_dataset_fix_label = []

    self.test_dataset = []
    self.test_dataset_label = []
    self.test_dataset_fix_label = []

    self.attr2idx = { }
    self.idx2attr = { }

    all_attr_names = self.lines[1].split()
    for i, attr_name in enumerate( all_attr_names ):
        self.attr2idx[attr_name] = i
        self.idx2attr[i] = attr_name

    lines = self.lines[2:]
    random.seed( 1234 )
    random.shuffle( lines )

    for i, line in enumerate( lines ):
        split = line.split()
        filename = os.path.join( self.data_path, split[0] )
        values = split[1:]

        label = []

        for attr_name in self.selected_attrs:
            idx = self.attr2idx[attr_name]

            if values[idx] == '1':
                label.append( 1 )
            else:
                label.append( 0 )

        if i < 2000:
            self.test_dataset.append( filename )
            self.test_dataset_label.append( label )
        else:
            self.train_dataset.append( filename )
            self.train_dataset_label.append( label )
        # ['./dataset/celebA/train/019932.jpg', [1, 0, 0, 0, 1]]

    self.test_dataset_fix_label = create_labels( self.test_dataset_label, self.selected_attrs )
    self.train_dataset_fix_label = create_labels( self.train_dataset_label, self.selected_attrs )

    print( '\n Finished preprocessing the Diffuse dataset...' )


def create_labels( c_org, selected_attrs = None ):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    c_org = np.asarray( c_org )
    hair_color_indices = []
    for i, attr_name in enumerate( selected_attrs ):
        if attr_name in ['0deg', '45deg', '90deg', '135deg', 'est_diffuse']:
            hair_color_indices.append( i )

    c_trg_list = []

    for i in range( len( selected_attrs ) ):
        c_trg = c_org.copy()

        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1.0
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0.0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append( c_trg )

    c_trg_list = np.transpose( c_trg_list, axes = [1, 0, 2] )  # [c_dim, bs, ch]

    return c_trg_list


# ----------------------------------------
def train( generator, discriminator, gan_model, latent_dim ):
    # Main training model
    return


def check_gpu():
    # ----------------------------------------
    # SETUP GPU
    # ----------------------------------------
    # # Testing and enabling GPU
    print( tf.test.is_built_with_cuda() )
    print( "Num GPUs Available: ", len( tf.config.list_physical_devices( 'GPU' ) ) )
    tf.config.list_physical_devices( 'GPU' )

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession( config = config )


# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------------
# ================= MAIN =================
# ----------------------------------------
def main():
    # To setup the various variables for the network
    args = parse_args()

    # check gpu availability
    check_gpu()

    # Load the dataset with resized polar and estimated diffuse images
    OriginalImageStack, img_0deg, img_45deg, img_90deg, img_135deg, estimated_diffuse = load_dataset( args )
    print( "[LOADED IMAGE] - Dataset size: ", len( img_0deg ), "images" )

    image_size = args.image_resize
    # ----------------------------------------
    # create the generator
    G = define_generator( args )
    # summarize the model
    G.summary()
    # ----------------------------------------
    # create discriminator
    D = define_descriminator( args )
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

    # Compute output for gradient penalty.
    rd_avg = RandomWeightedAverage()
    rd_avg.define_batch_size( args.batch_size )
    x_hat = rd_avg( [x_real, x_fake] )
    out_src, _ = D( x_hat )

    # Use Python partial to provide loss function with additional 'averaged_samples' argument
    partial_gp_loss = partial( gradient_penalty_loss, averaged_samples = x_hat )
    partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

    # Define training model D
    train_D = Model( [x_real, label_trg], [out_src_real, out_cls_real, out_src_fake, out_src] )

    # Setup loss for train_D
    train_D.compile( loss = [wasserstein_loss, classification_loss, wasserstein_loss, partial_gp_loss],
                     optimizer = Adam( lr = args.d_lr, beta_1 = args.beta1, beta_2 = args.beta2 ),
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
                     optimizer = Adam( lr = args.g_lr, beta_1 = args.beta1, beta_2 = args.beta2 ),
                     loss_weights = [1, args.lambda_cls, args.lambda_rec] )

    """ Input Image"""
    Image_data_class = ImageData( data_dir = args.data_dir, selected_attrs = args.selected_attrs )
    Image_data_class.preprocess()


# ----------------------------------
if __name__ == '__main__':
    main()