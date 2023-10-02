"""
# -----------------------------------------------------------
SHMGAN -  Removal of Specular Highlights by a GAN Network


Uses Packages:
    Python 3.8
    CUDA 11.8
    cuDnn 8.0
    Tensorflow 2.8 + Keras 2.8
    Tensorflow addons 0.17.1

(C) 2023 Atif Anwer, INSA Rouen, France
Email: atif.anwer@insa-rouen.fr

Folder Structure expected for the network to run:
├── PolarizedSource
    ├── ED       // Estimated Diffuse folder pre-populated (for now)
    ├── I0       // Polar RGB images
    ├── I135
    ├── I45
    └── I90

# -----------------------------------------------------------
"""
import gc
import os
import time

# from PIL import Image
from timeit import default_timer

import tensorflow as tf
from keras.optimizers import adam_v2
from matplotlib import pyplot as plt
from tensorflow.python.keras import (
    regularizers,
)
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import (
    AveragePooling2D,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    MaxPooling2D,
)
from tensorflow.python.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

from datasetLoader import datasetLoad

# Import SpecSeg network
from SpecSeg import SpecSeg
from utils import (
    plot_single_image,
    printProgressBar,
    rescale_01,
)

print(  "------------------------------------",
        "\nTensoorflow version:", tf.__version__,
        "\nKeras Version", tf.keras.__version__ ,
        "\n------------------------------------",)


# Removes error when running Tensorflow on GPU
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# Reduces Tensorflow messages other than errors or important messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# To debug @tf.functions in vscode (otherwise breakpoints dont work)
# tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


# ------------------------------------------------
# =============== THE MAIN CLASS ===============
# ------------------------------------------------
class ShmGANwithSSpecSeg( object ):
    # ------------------------------------------------
    #
    # ██╗███╗   ██╗██╗████████╗
    # ██║████╗  ██║██║╚══██╔══╝
    # ██║██╔██╗ ██║██║   ██║
    # ██║██║╚██╗██║██║   ██║
    # ██║██║ ╚████║██║   ██║
    # ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝
    # ------------------------------------------------
    def __init__( self, args ):

        # only create Comet experiment if training, not testing
        # Create an experiment with your api key
        # if args.mode == 'train':
        #     self.comet_experiment = Experiment(
        #         api_key                            = "insert_your_api_key_here",
        #         project_name                       = "insert_your_project_name_here",
        #         workspace                          = "insert_your_workspace_here",
        #         auto_param_logging                 = True,
        #         auto_metric_logging                = True,
        #         log_env_details                    = True,
        #         log_code                           = True,                        # code logging
        #         log_graph                          = True,
        #         log_env_gpu                        = True,
        #         log_env_host                       = True,
        #         log_env_cpu                        = True,
        #         auto_histogram_tensorboard_logging = True,
        #         auto_histogram_weight_logging      = True,
        #         auto_histogram_gradient_logging    = True,
        #         auto_histogram_activation_logging  = True,
        #         # auto_histogram_epoch_rate=1,
        #     )
        #     self.comet_experiment.add_tag("insert_your_tags_here")

        # Model configuration.
        self.c_dim        = args.c_dim
        self.image_size   = args.image_size  # the size of the image after resizing

        # Training configuration.
        self.batch_size          = args.batch_size  # batch size for training
        self.num_epochs          = args.num_epochs
        self.num_iteration_decay = args.num_iteration_decay
        self.g_lr                = args.g_lr
        self.d_lr                = args.d_lr
        self.n_critic            = args.n_critic
        self.beta1               = args.beta1
        self.beta2               = args.beta2
        self.d_repeat_num        = args.d_repeat_num

        # Test configurations.
        # self.test_iters = args.test_iters

        # Miscellaneous.
        self.mode = args.mode

        # Directories.
        self.data_dir            = args.data_dir  # the root folder containing polarimetric sub-folders
        self.model_save_dir      = args.model_save_dir
        self.checkpoint_save_dir = args.checkpoint_save_dir
        self.result_dir          = args.result_dir
        self.log_dir             = args.log_dir
        # self.lambda_recons       = args.lambda_recons
        # self.lambda_class        = args.lambda_class

        # Step size.
        self.log_step             = args.log_step
        self.checkpoint_save_step = args.checkpoint_save_step

        # Misc parameters
        self.filter_size  = args.filter_size
        self.seed         = 25
        self.randomness   = 0.50
        self.dropout_amnt = 0.2  # ( 0.2 used in CollaGAN)

        self.TARGET_LABELS = 0.90  # - Label smoothing by not using hard 1.0 value
        # To use LSGAN
        self.use_lsgan = True

        self._graph=tf.Graph()

        # using different optimizers for G and D
        # To use decayed learning rate, replace the LR with this
        decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(   initial_learning_rate = self.g_lr, \
                                                                        decay_steps = 10000, decay_rate = 00.95, \
                                                                        staircase = False )

        self.optimizer_G = adam_v2.Adam( learning_rate = decayed_lr, beta_1 = self.beta1, beta_2 = self.beta2 )
        self.optimizer_D = adam_v2.Adam( learning_rate = decayed_lr, beta_1 = self.beta1, beta_2 = self.beta2 )
        self.optimizer_SpecSeg = adam_v2.Adam( learning_rate = decayed_lr, beta_1 = self.beta1, beta_2 = self.beta2 )

        # Use this only if Mixed precision = mixed_float16 is used. Otherwise dont use LossScaleOptimizer
        # https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer
        # mixed_precision.set_global_policy('mixed_float16')
        # self.optimizer_G = mixed_precision.LossScaleOptimizer( self.optimizer_G )
        # self.optimizer_D = mixed_precision.LossScaleOptimizer( self.optimizer_D )

        """
        Keep or delete old checkpoints
        """
        # self.delete_old_checkpoints = args.delete_old_checkpoints
        self.delete_old_checkpoints = False

        # Only traing Discriminator at first. Train Generator after n epochs
        self.train_G_after = 0

        self.c_dim        = 5
        self.g_conv_dim   = 64
        self.g_repeat_num = 6

        # for plotting histograms of gradients
        self.gradmapD = {}
        self.gradmapG = {}

        self.init = RandomNormal(mean=0.0, stddev=0.02, seed=42)     # suggested by DCGAN

        # Initialize for use in training
        self.random_flip = 0.0

        # initializing the specular candidate matrix
        self.specular_candidate = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        # Trainable weights (?)
        self.gamma = tf.Variable(initial_value=0, shape=(), trainable=True, dtype=tf.float32, name="Gamma")

        # Zhao et al (2018) Loss function alpha as defined in paper
        self.alpha = 0.84


    # ------------------------------------------------
    #
    #  ██████  ███████ ███    ██ ███████ ██████   █████  ████████  ██████  ██████
    # ██       ██      ████   ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    # ██   ███ █████   ██ ██  ██ █████   ██████  ███████    ██    ██    ██ ██████
    # ██    ██ ██      ██  ██ ██ ██      ██   ██ ██   ██    ██    ██    ██ ██   ██
    #  ██████  ███████ ██   ████ ███████ ██   ██ ██   ██    ██     ██████  ██   ██
    #
    # Generator has two inputs and one output
    # INPUT: Concatenated multiple 5x Y-channel images (with concatenated label channels)
    # OUTPUT: Single Image (Single Y channel only - concatenate with CbCr after generation) (generated_image)
    # ------------------------------------------------
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def build_generator( self ):
        # inp_images = INPUT2 >> Concatenated 5x Y Channel images
        inp_images = Input( shape = (self.image_size, self.image_size, 10) )

        # UNET Architecture inspired from CollaGAN
        # inp                                        o/p
        # └── d1 -------------------------------- u4 ──┘
        #     └── d2 ----------------------- u3 ──┘
        #         └── d3 -------------- u2 ──┘
        #             └── d4 ----- u1 ──┘
        #                   └─ im ─┘

        # DOWNSAMPLE 5 times
        N = self.filter_size
        x = inp_images
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
            # x = LeakyReLU()( x )
        down1 = x
        self.attn_1, pooled = self.attention_layer( spec=self.specular_candidate, filter_size=N, pool=False )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d1

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        down2 = x
        self.attn_2, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d2

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        down3 = x
        self.attn_3, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = 'same' )( x )
        N *= 2
        # ----d3

        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        down4 = x
        self.attn_4, pooled = self.attention_layer( spec=pooled, filter_size=N, pool=True )
        x = AveragePooling2D( pool_size = (2, 2), strides = None, padding = "same" )( x )
        # ----d4

        # Adding 1x1 conv layers
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 1, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        #  ----- 1x1 layers
        # N *= 2

        # Multiply the skip connection with the attention layer generated from the Mask.
        # Note that attention layer is between [0,1] after sigmoid
        # Note that the first call does not Maxpool the mask to retain dimension consistency

        down4 = down4 + self.attn_4
        down3 = down3 + self.attn_3
        down2 = down2 + self.attn_2
        down1 = down1 + self.attn_1

        # # UPSAMPLE
        # ----u1
        # N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu  )( x )
        x = Concatenate()( [x, down4] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init )( x )
        # ----u2
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu  )( x )
        x = Concatenate()( [x, down3] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        # ----u3
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu  )( x )
        x = Concatenate()( [x, down2] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        # ----u4
        N /= 2
        x = Conv2DTranspose( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu  )( x )
        x = Concatenate()( [x, down1] )
        for i in range( 2 ):
            x = Conv2D( filters = N, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( x )
            x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )

        # Output is a single Y channel image
        genOutput = Conv2D( filters = 1, kernel_size = 1, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu)( x )
        return Model( inp_images, genOutput, name = 'SHM_Generator' )

    # ------------------------------------------------
    #
    # ██████  ██ ███████  ██████ ██████  ██ ███    ███ ██ ███    ██  █████  ████████  ██████  ██████
    # ██   ██ ██ ██      ██      ██   ██ ██ ████  ████ ██ ████   ██ ██   ██    ██    ██    ██ ██   ██
    # ██   ██ ██ ███████ ██      ██████  ██ ██ ████ ██ ██ ██ ██  ██ ███████    ██    ██    ██ ██████
    # ██   ██ ██      ██ ██      ██   ██ ██ ██  ██  ██ ██ ██  ██ ██ ██   ██    ██    ██    ██ ██   ██
    # ██████  ██ ███████  ██████ ██   ██ ██ ██      ██ ██ ██   ████ ██   ██    ██     ██████  ██   ██
    #
    # Discriminator has One inputs and two outputs
    # INPUT: Image to discriminate      (inp_img)
    # OUTOUT1: Real or Fake             (out_RealFake)
    # OUTPUT2: Category of the image    (out_cat_label)
    # ------------------------------------------------
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def build_discriminator( self ):
        """Discriminator network with PatchGAN - taken from StarGAN implementation."""
        # NOTE: Input image to discriminator is RGB after concatenating with Y with CbCr
        # so 3 channels are only required without any labels concatenated
        inp_img = Input( shape = (self.image_size, self.image_size, 3) )

        x = inp_img
        N = self.filter_size  # default = 64... Should always be image_size/2 (?)

        x = GaussianNoise( 0.1 )(x)
        x = self.Conv_LReLU_IN ( x, N )     #64
        x = self.Conv_LReLU_IN ( x, N * 2)  #128
        x = self.Conv_LReLU_IN ( x, N * 4)  #256
        x = self.Conv_LReLU_IN ( x, N * 8)  #512

        attn_disc, _ = self.attention_layer( spec=self.specular_candidate, filter_size=N * 8, pool=True, poolsize=(16,16) )
        x = x + attn_disc

        x = self.Conv_LReLU_IN ( x, N * 16) #1024

        x = Dropout( rate=self.dropout_amnt )( x )

        out_RealFake = Conv2D(  filters = 1, kernel_size = 3, strides = 1, padding = 'same', \
                                use_bias           = False,            \
                                activation         = tf.nn.leaky_relu, \
                                dtype              = tf.float32,       \
                                kernel_initializer = self.init  )( x )

        FC_Layer = Flatten()(x)
        out_cat_label = Dense(  units              = self.c_dim, \
                                use_bias           = False,     \
                                kernel_initializer = self.init, \
                                dtype              = tf.float32 )(FC_Layer)        # (None, 5)

        # The model is one input and two output (Real/Fake and its class)
        D = Model( inp_img, [out_RealFake, out_cat_label], name = 'SHM_Discriminator' )

        return D

    # ----------------------------------------------
    # Define the CNL block as in CollaGAN
    # Conv2d + InstNorm + LeakyRELU called 2 times
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def Conv_LReLU_IN( self , x, N ):
        x = Conv2D( filters = N, kernel_size = 3, strides = 2, padding = "same", kernel_initializer=self.init, use_bias=False, kernel_regularizer=regularizers.l2(0.001), activation=tf.nn.leaky_relu, dtype=tf.float32 )( x )
        x = InstanceNormalization( axis = -1, epsilon=0.000001, center=True,  beta_initializer=self.init  )( x )
        return x

    # ----------------------------------------------
    #
    #  █████  ████████ ████████ ███████ ███    ██ ████████ ██  ██████  ███    ██
    # ██   ██    ██       ██    ██      ████   ██    ██    ██ ██    ██ ████   ██
    # ███████    ██       ██    █████   ██ ██  ██    ██    ██ ██    ██ ██ ██  ██
    # ██   ██    ██       ██    ██      ██  ██ ██    ██    ██ ██    ██ ██  ██ ██
    # ██   ██    ██       ██    ███████ ██   ████    ██    ██  ██████  ██   ████
    #
    # ----------------------------------------------
    # Generating the attention layer from the specular candidate from Shen function
    # Note that the filter size is provided on call
    # NOTE: Sigmoid activation included in the layer.
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def attention_layer( self, spec, filter_size, poolsize=(2,2) ,pool=True ):
        if pool is True:
            pooled = MaxPooling2D( pool_size = poolsize, strides = None, padding = "same" )( spec )
        else:
            pooled = self.specular_candidate

        spec = Conv2D( filters = filter_size, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( pooled )
        spec = Conv2D( filters = filter_size, kernel_size = 3, strides = 1, padding = "same", kernel_initializer=self.init, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(0.001) )( spec )
        return spec, pooled

    # Sauce: https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    # Sauce: https://www-cxybb-com.translate.goog/article/qq_35586657/98875077?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=nui,sc
    # Attention Mecanism from SAGAN paper. Calculates attention
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def sagan_attention( self, input_tensor, filter_size, poolsize=(2,2), pool=True ):

        if pool is True:
            input_tensor = MaxPooling2D( pool_size = poolsize, strides = None, padding = "same" )( input_tensor )

        f = Conv2D( filters = filter_size // 8, kernel_size = 1, padding = "same", kernel_initializer=self.init )( input_tensor )
        g = Conv2D( filters = filter_size // 8, kernel_size = 1, padding = "same", kernel_initializer=self.init )( input_tensor )
        h = Conv2D( filters = filter_size, kernel_size = 1, padding = "same", kernel_initializer=self.init )( input_tensor )

        input_shape = f.shape.as_list()
        input_shape[0]
        height      = input_shape[1]
        width       = input_shape[2]
        channels    = filter_size // 8

        f = tf.reshape(f, shape = [ height*width, channels] )
        g = tf.reshape(g, shape = [ height*width, channels] )
        h = tf.reshape(h, shape = [ height*width, filter_size] )

        energy = tf.matmul( g, f, transpose_b=True)
        # Note: Original SAGAN uses Softmax
        attention = tf.nn.softmax( energy )
        out = tf.matmul( attention, h )

        gamma = self.gamma

        # shape = input_tensor.get_shape().as_list()
        # shape = shape[1:]
        out = tf.reshape( out, shape= [ height , width, filter_size] )

        output = gamma * out + input_tensor

        return output, attention


    """
    # ------------------------------------------------
    #
    # ████████ ██████   █████  ██ ███    ██       ███████ ████████ ███████ ██████
    #    ██    ██   ██ ██   ██ ██ ████   ██       ██         ██    ██      ██   ██
    #    ██    ██████  ███████ ██ ██ ██  ██ █████ ███████    ██    █████   ██████
    #    ██    ██   ██ ██   ██ ██ ██  ██ ██            ██    ██    ██      ██
    #    ██    ██   ██ ██   ██ ██ ██   ████       ███████    ██    ███████ ██
    #
    #                   BUILD THE MODEL
    # ------------------------------------------------
    """
    # jit_compile=True
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def train_step( self, orig0, orig45, orig90, orig135, origED ):

        # create zeros and ones for labels
        tmp_zeros = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )
        trg_ones  = tf.ones( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        self.target_label_0deg   = tf.Variable( [self.TARGET_LABELS,0,0,0,0],  dtype=tf.float32 )
        self.target_label_45deg  = tf.Variable( [0,self.TARGET_LABELS,0,0,0],  dtype=tf.float32 )
        self.target_label_90deg  = tf.Variable( [0,0,self.TARGET_LABELS,0,0],  dtype=tf.float32 )
        self.target_label_135deg = tf.Variable( [0,0,0,self.TARGET_LABELS,0],  dtype=tf.float32 )
        self.target_label_ED     = tf.Variable( [0,0,0,0,self.TARGET_LABELS],  dtype=tf.float32 )


        ds1_yuv = self.custom_per_image_standardization( tf.image.rgb_to_yuv( orig0[:, :, :, :] ) )
        ds2_yuv = self.custom_per_image_standardization( tf.image.rgb_to_yuv( orig45[:, :, :, :] ) )
        ds3_yuv = self.custom_per_image_standardization( tf.image.rgb_to_yuv( orig90[:, :, :, :] ) )
        ds4_yuv = self.custom_per_image_standardization( tf.image.rgb_to_yuv( orig135[:, :, :, :] ) )
        ds5_yuv = self.custom_per_image_standardization( tf.image.rgb_to_yuv( origED[:, :, :, :] ) )

        I0_Ych   = ds1_yuv[:, :, :, 0, tf.newaxis]
        I45_Ych  = ds2_yuv[:, :, :, 0, tf.newaxis]
        I90_Ych  = ds3_yuv[:, :, :, 0, tf.newaxis]
        I135_Ych = ds4_yuv[:, :, :, 0, tf.newaxis]
        IED_Ych  = ds5_yuv[:, :, :, 0, tf.newaxis]
        # predict outside gradient tape
        self.specular_candidate = (self.SpecSeg.predict(I90_Ych, verbose=0))

        # using separate tapes for both D and G (sauce: pix2pix tutorial from TF site)
        with tf.GradientTape( persistent=True ) as tape:

            # NOTE: Generator is not trainiable in the first step, but D is trainable
            self.G.trainable = False
            self.D.trainable = True

            """--------------------G(1)-------------------"""
            # STEP1: Generate Fake images from the Generator

            # averaging the CB and CR channels to get the estimated CbCr to use later
            averageCbCr = ( ds1_yuv[:, :, :, 1:] + ds2_yuv[:, :, :, 1:] + ds3_yuv[:, :, :, 1:] + ds4_yuv[:, :, :, 1:] + ds5_yuv[:, :, :, 1:] ) / 5.0
            averageCbCr = tf.cast(averageCbCr, tf.float32)

            # generating random numbers
            RNG1 = tf.random.uniform( [] ) < self.randomness
            RNG2 = tf.random.uniform( [] ) < self.randomness
            RNG3 = tf.random.uniform( [] ) < self.randomness
            RNG4 = tf.random.uniform( [] ) < self.randomness
            RNG5 = tf.random.uniform( [] ) < self.randomness

            # Generate inputs (Zero or orig image y ch) randomly for the training process
            # Extracting the Y channels in a [1,rows,cols,5] stack
            rand_inp1 = tf.cond( RNG1, lambda: tmp_zeros, lambda: I0_Ych   )
            rand_inp2 = tf.cond( RNG2, lambda: tmp_zeros, lambda: I45_Ych  )
            rand_inp3 = tf.cond( RNG3, lambda: tmp_zeros, lambda: I90_Ych  )
            rand_inp4 = tf.cond( RNG4, lambda: tmp_zeros, lambda: I135_Ych )
            rand_inp5 = tf.cond( RNG5, lambda: tmp_zeros, lambda: IED_Ych  )

            # plt.imshow(tf.squeeze( ds1[:, :, :, 0, tf.newaxis] ), cmap=cm.gray)       - Works

            rand_input_Ych = tf.concat( [rand_inp1, rand_inp2, rand_inp3, rand_inp4, rand_inp5], axis = 3 )

            # Generate a random number between 1 and 5 (maxval=6 required not 5 )
            tf.random.uniform(shape=(), minval=1, maxval=6, dtype=tf.int32)

            # making the G1 generate ED everytime ...
            self.gen_input          = tf.concat( [rand_input_Ych, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones], axis = 3 )
            self.target_img         = origED
            self.Target_angle_label = self.target_label_ED

            # debug_plot( self.gen_input )

            # Generate only 1 Y channel image, based on random target class.
            self.gen_Y   = self.G ( self.gen_input, training=False )

            """-------------------D(1)--------------------"""
            # STEP2: Discriminate the Fake generated image and the Class of the fake image

            # concatenate the generated Y channel with the average CbCr to get the complete Generated YCbCr image
            self.gen_YCbCr   = tf.concat( [self.gen_Y, averageCbCr], axis = 3 )

            # plot_single_image (  self.gen_YCbCr )
            # Normalize using std. ev
            avg_stddev_arr      = tf.reduce_mean(self.stddev_arr)
            tf.reduce_mean(self.mean_arr)
            self.gen_rgb_output = tf.image.yuv_to_rgb( ( tf.multiply(self.gen_YCbCr, avg_stddev_arr) ) * 255.0 )

            # convert to RGB for the discriminator
            self.gen_rgb   = tf.image.yuv_to_rgb( self.gen_YCbCr )

            # plot_single_image ( self.gen_Y )
            # plot_single_image ( self.gen_rgb )

            # DALEK: Discriminate!!
            self.RealFake_gen_D1  , self.label_gen_D1   = self.D( self.gen_rgb, training=True )

            """-------------------D(2)--------------------"""
            # STEP3: Discriminate the Target images and the Class of the target image
            self.RealFake_target_D2  , self.label_target_D2   = self.D( self.target_img, training=True )


            """--------------------G(2)-------------------"""
            # STEP5: Generate Fake images from the Generator

            # NOTE: Generator is Now trainiable in the second step, but D is not
            self.G.trainable = True
            self.D.trainable = False


            # Using previous Random values, re-substitute the generated images for cyclic consistency
            # note that the dsx[] are the original batch of images passed
            rand_inpA = tf.cond( RNG1, lambda: self.gen_Y, lambda: I0_Ych   )
            rand_inpB = tf.cond( RNG2, lambda: self.gen_Y, lambda: I45_Ych  )
            rand_inpC = tf.cond( RNG3, lambda: self.gen_Y, lambda: I90_Ych  )
            rand_inpD = tf.cond( RNG4, lambda: self.gen_Y, lambda: I135_Ych )
            rand_inpE = tf.cond( RNG5, lambda: self.gen_Y, lambda: IED_Ych  )

            # append the randomly generated inputs to each channel. The zeros define the channel to be generated
            cyc_Y1 = tf.concat( [tmp_zeros, rand_inpB, rand_inpC, rand_inpD, rand_inpE], axis = 3 )
            cyc_Y2 = tf.concat( [rand_inpA, tmp_zeros, rand_inpC, rand_inpD, rand_inpE], axis = 3 )
            cyc_Y3 = tf.concat( [rand_inpA, rand_inpB, tmp_zeros, rand_inpD, rand_inpE], axis = 3 )
            cyc_Y4 = tf.concat( [rand_inpA, rand_inpB, rand_inpC, tmp_zeros, rand_inpE], axis = 3 )
            cyc_Y5 = tf.concat( [rand_inpA, rand_inpB, rand_inpC, rand_inpD, tmp_zeros], axis = 3 )

            # Append one-hot tensors as labels
            cyclic_input1 = tf.concat( [cyc_Y1, trg_ones, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input2 = tf.concat( [cyc_Y2, tmp_zeros, trg_ones, tmp_zeros, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input3 = tf.concat( [cyc_Y3, tmp_zeros, tmp_zeros, trg_ones, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input4 = tf.concat( [cyc_Y4, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones, tmp_zeros], axis = 3 )
            cyclic_input5 = tf.concat( [cyc_Y5, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones], axis = 3 )

            # debug_plot( cyclic_input1 )
            # debug_plot( cyclic_input2 )
            # debug_plot( cyclic_input3 )
            # debug_plot( cyclic_input4 )
            # debug_plot( cyclic_input5 )

            # Generate 5 Y channel CYCLIC images, to reconstruct the original images.
            cyc_0_Y   = self.G( cyclic_input1, training=True )
            cyc_45_Y  = self.G( cyclic_input2, training=True )
            cyc_90_Y  = self.G( cyclic_input3, training=True )
            cyc_135_Y = self.G( cyclic_input4, training=True )
            cyc_ED_Y  = self.G( cyclic_input5, training=True )

            """--------------------D(3)-------------------"""
            # STEP3: ?

            # concatenate the generated CYCLIC Y channel with the average CbCr to get the complete Generated YCbCr image
            cyc_gen0_yuv   = tf.concat( [ cyc_0_Y, averageCbCr], axis = 3 )
            cyc_gen45_yuv  = tf.concat( [ cyc_45_Y, averageCbCr], axis = 3 )
            cyc_gen90_yuv  = tf.concat( [ cyc_90_Y, averageCbCr], axis = 3 )
            cyc_gen135_yuv = tf.concat( [ cyc_135_Y, averageCbCr], axis = 3 )
            cyc_genED_yuv  = tf.concat( [ cyc_ED_Y, averageCbCr], axis = 3 )

            # convert to RGB for the discriminator
            self.cyc_gen0_rgb   = tf.image.yuv_to_rgb( cyc_gen0_yuv )
            self.cyc_gen45_rgb  = tf.image.yuv_to_rgb( cyc_gen45_yuv )
            self.cyc_gen90_rgb  = tf.image.yuv_to_rgb( cyc_gen90_yuv )
            self.cyc_gen135_rgb = tf.image.yuv_to_rgb( cyc_gen135_yuv )
            self.cyc_genED_rgb  = tf.image.yuv_to_rgb( cyc_genED_yuv )

            # Train discriminator to differentiate the generated CYCLIC images
            RealFake_cyc_gen0_D3  , label_cyc_gen0_D3   = self.D( self.cyc_gen0_rgb, training=False )
            RealFake_cyc_gen45_D3 , label_cyc_gen45_D3  = self.D( self.cyc_gen45_rgb, training=False )
            RealFake_cyc_gen90_D3 , label_cyc_gen90_D3  = self.D( self.cyc_gen90_rgb, training=False )
            RealFake_cyc_gen135_D3, label_cyc_gen135_D3 = self.D( self.cyc_gen135_rgb, training=False )
            RealFake_cyc_genED_D3 , label_cyc_genED_D3  = self.D( self.cyc_genED_rgb, training=False )

            """-------------------D(4)--------------------"""
            # STEP1: Train Discriminator with real images only
            # Discriminator outs for each of the images
            # Discriminator outputs whether image was real/fake AND the type of the image
            # REAL or FAKE , POLAR Ө   = discriminator( real_img )
            RealFake_orig0_D4   , self.label_orig0_D4   = self.D( orig0, training=False )
            RealFake_orig45_D4  , self.label_orig45_D4  = self.D( orig45, training=False )
            RealFake_orig90_D4  , self.label_orig90_D4  = self.D( orig90, training=False )
            RealFake_orig135_D4 , self.label_orig135_D4 = self.D( orig135, training=False )
            RealFake_origED_D4  , self.label_origED_D4  = self.D( origED, training=False )

            # ==================================================================
            #
            # ██       ██████  ███████ ███████ ███████ ███████
            # ██      ██    ██ ██      ██      ██      ██
            # ██      ██    ██ ███████ ███████ █████   ███████
            # ██      ██    ██      ██      ██ ██           ██
            # ███████  ██████  ███████ ███████ ███████ ███████
            # ==================================================================

            """----------------Losses--------------------"""
            """# STEP4 A: Generator GAN loss"""
            # Total 4 losses (CollaGAN) or 4 (SHMGAN?)
            # 1. Generator      - 2x Genreator Loss (Real/Fake Image)
            #                   -
            # 2. Discriminator  - 2x Classification Loss (Classified Labels)
            #
            # The loss is to compare the generated image from its actual targeted image.
            # There will be 5 cyclic losses, one for each image since all 5 images are being generated
            # real -> generated -> cyclic

            # L1_loss = tf.keras.losses.MeanSquaredError()

            # GENERATOR LOSS

            # aka G_gan_loss_cyc
            D3_RealFake_cyc1 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_cyc_gen0_D3, self.TARGET_LABELS ) )
            D3_RealFake_cyc2 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_cyc_gen45_D3, self.TARGET_LABELS ) )
            D3_RealFake_cyc3 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_cyc_gen90_D3, self.TARGET_LABELS ) )
            D3_RealFake_cyc4 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_cyc_gen135_D3, self.TARGET_LABELS ) )
            D3_RealFake_cyc5 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_cyc_genED_D3, self.TARGET_LABELS ) )
            self.D3_RealFake_cyc = D3_RealFake_cyc1 + D3_RealFake_cyc2 + D3_RealFake_cyc3 + D3_RealFake_cyc4 + D3_RealFake_cyc5

            # Calculating loss for the generated image
            self.D1_RealFake_loss = tf.math.reduce_mean( tf.math.squared_difference( self.RealFake_gen_D1, self.TARGET_LABELS ) )

            self.G_gan_loss = ( self.D3_RealFake_cyc + self.D1_RealFake_loss ) / 6.0
            # -------------------------------------------------

            # oneHot lables will be of shape (1, 5)
            oneHot_lbl1 = tf.reshape(tf.one_hot(tf.cast(0,tf.uint8),5),[1,5])
            oneHot_lbl2 = tf.reshape(tf.one_hot(tf.cast(1,tf.uint8),5),[1,5])
            oneHot_lbl3 = tf.reshape(tf.one_hot(tf.cast(2,tf.uint8),5),[1,5])
            oneHot_lbl4 = tf.reshape(tf.one_hot(tf.cast(3,tf.uint8),5),[1,5])
            oneHot_lbl5 = tf.reshape(tf.one_hot(tf.cast(4,tf.uint8),5),[1,5])
            oneHot_lbl_target = tf.reshape(self.Target_angle_label,[1,5])

            # NOTE: Warning: This op EXPECTS unscaled logits, since it performs a softmax on logits internally for efficiency.
            # Do not call this op with the output of softmax, as it will produce incorrect results.
            # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
            # Binary crossentropy = Sigmoid crossentropy
            # Categorical crossentropy = Softmax crossentropy (for multi-class classification)
            D3_classification_loss1 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl1, logits = label_cyc_gen0_D3   )
            D3_classification_loss2 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl2, logits = label_cyc_gen45_D3  )
            D3_classification_loss3 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl3, logits = label_cyc_gen90_D3  )
            D3_classification_loss4 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl4, logits = label_cyc_gen135_D3 )
            D3_classification_loss5 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl5, logits = label_cyc_genED_D3  )
            self.D3_classification_loss = D3_classification_loss1 + D3_classification_loss2 + D3_classification_loss3 + D3_classification_loss4 + D3_classification_loss5

            self.D1_classification_loss = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl_target, logits = self.label_gen_D1, name="error_orig" )

            self.G_clsf_loss = ( self.D3_classification_loss + self.D1_classification_loss ) / 6.0
            # -------------------------------------------------

            # DISCRIMINATOR LOSS:

            D4_classification_loss1 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl1, logits = self.label_orig0_D4   )
            D4_classification_loss2 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl2, logits = self.label_orig45_D4  )
            D4_classification_loss3 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl3, logits = self.label_orig90_D4  )
            D4_classification_loss4 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl4, logits = self.label_orig135_D4 )
            D4_classification_loss5 = tf.nn.softmax_cross_entropy_with_logits( labels = oneHot_lbl5, logits = self.label_origED_D4  )
            self.D4_classification_loss = D4_classification_loss1 + D4_classification_loss2 + D4_classification_loss3 + D4_classification_loss4 + D4_classification_loss5

            # The Least Squares Generative Adversarial Network, or LSGAN for short, is an extension to the GAN architecture that addresses the problem of vanishing gradients and loss saturation.
            # Using LSGAN for D2 and D4 outputs:
            # D_loss = 0.5 * tf.reduce_mean((D_real-1)^2) + tf.reduce_mean(D_fake^2)
            # G_loss = 0.5 * tf.reduce_mean((D_fake -1)^2)
            #  D2 + D1
            self.D2_RealFake_target = tf.math.reduce_mean( tf.math.squared_difference( self.RealFake_target_D2 , self.TARGET_LABELS )) + tf.math.reduce_mean( tf.math.square( self.RealFake_gen_D1 ) )
            # D4 + D3
            D4_1 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig0_D4 , self.TARGET_LABELS ))   + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen0_D3 ) )
            D4_2 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig45_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen45_D3 ) )
            D4_3 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig90_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen90_D3 ) )
            D4_4 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_orig135_D4 , self.TARGET_LABELS )) + tf.math.reduce_mean( tf.math.square( RealFake_cyc_gen135_D3 ) )
            D4_5 = tf.math.reduce_mean( tf.math.squared_difference( RealFake_origED_D4 , self.TARGET_LABELS ))  + tf.math.reduce_mean( tf.math.square( RealFake_cyc_genED_D3 ) )
            self.D4_RealFake_cyc = D4_1 + D4_2 + D4_3 + D4_4 + D4_5 + self.D2_RealFake_target

            (self.D4_RealFake_cyc)/6.0 + (self.D4_classification_loss)/5.0

            # --------------------      L1 Loss       --------------------
            # # Adding L1 loss for generator to force image generation to look like the original images
            # L1_loss = tf.keras.losses.MeanSquaredError()
            # L1_loss_G1   = L1_loss(self.gen_rgb, self.target_img )
            # L1_loss_G2_1 = L1_loss(self.cyc_gen0_rgb, orig0 )
            # L1_loss_G2_2 = L1_loss(self.cyc_gen45_rgb, orig45 )
            # L1_loss_G2_3 = L1_loss(self.cyc_gen90_rgb, orig90 )
            # L1_loss_G2_4 = L1_loss(self.cyc_gen135_rgb, orig135 )
            # L1_loss_G2_5 = L1_loss(self.cyc_genED_rgb, origED )

            #  The cycle consistency loss is defined as the sum of the L1 distances between the real images from each domain and their generated counterparts.
            # Source: https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation/blob/master/losses.py
            L1_loss_G1    = tf.reduce_mean( tf.abs( self.gen_rgb        - self.target_img ))
            L1_loss_G2_1  = tf.reduce_mean( tf.abs( self.cyc_gen0_rgb   - orig0 ))
            L1_loss_G2_2  = tf.reduce_mean( tf.abs( self.cyc_gen45_rgb  - orig45 ))
            L1_loss_G2_3  = tf.reduce_mean( tf.abs( self.cyc_gen90_rgb  - orig90 ))
            L1_loss_G2_4  = tf.reduce_mean( tf.abs( self.cyc_gen135_rgb - orig135 ))
            L1_loss_G2_ED = tf.reduce_mean( tf.abs( self.cyc_genED_rgb  - origED ))

            self.L1_loss_Gen  = ( L1_loss_G2_1 + L1_loss_G2_2 + L1_loss_G2_3 + L1_loss_G2_4 + L1_loss_G1) / 5.0 + ( L1_loss_G2_ED ) * 10.0


            # --------------------      MS-SSIM       --------------------
            ## SSIM loss for generator
            # NOTE: SSIM is for greyscale only so using Y channel only
            # Using maxval= 5 from the collaGAN MIR paper
            # SSIM of YUV images (NOT RGB)
            ssim1 = tf.image.ssim ( rescale_01(cyc_gen0_yuv)   , rescale_01( ds1_yuv ) , 5 )
            ssim2 = tf.image.ssim ( rescale_01(cyc_gen45_yuv)  , rescale_01( ds2_yuv ) , 5 )
            ssim3 = tf.image.ssim ( rescale_01(cyc_gen90_yuv)  , rescale_01( ds3_yuv ) , 5 )
            ssim4 = tf.image.ssim ( rescale_01(cyc_gen135_yuv) , rescale_01( ds4_yuv ) , 5 )
            ssim5 = tf.image.ssim ( rescale_01(cyc_genED_yuv)  , rescale_01( ds5_yuv ) , 5 )

            # # RESCALING TO [0,1] BEFORE CALCULATING SSIM
            # power_factors=[(0.0448, 0.2856, 0.3001)]    # from TF documentation
            # # ssim1 = tf.image.ssim_multiscale ( img1 = rescale_01( cyc_gen0_yuv)   , img2 = rescale_01(ds1_yuv), max_val=1, power_factors = power_factors )
            # # ssim2 = tf.image.ssim_multiscale ( img1 = rescale_01( cyc_gen45_yuv)  , img2 = rescale_01(ds2_yuv), max_val=1, power_factors = power_factors )
            # # ssim3 = tf.image.ssim_multiscale ( img1 = rescale_01( cyc_gen90_yuv)  , img2 = rescale_01(ds3_yuv), max_val=1, power_factors = power_factors )
            # # ssim4 = tf.image.ssim_multiscale ( img1 = rescale_01( cyc_gen135_yuv) , img2 = rescale_01(ds4_yuv), max_val=1, power_factors = power_factors )
            # # ssim5 = tf.image.ssim_multiscale ( img1 = rescale_01( cyc_genED_yuv)  , img2 = rescale_01(ds5_yuv), max_val=1, power_factors = power_factors )
            # # self.ssim_cyc_loss  = (ssim1 + ssim2 + ssim3 + ssim4 + ssim5) / 5.0

            cyc_ssim_loss1 = tf.cond( RNG1 , lambda:0.0, lambda: -tf.math.log( ( 1.0 + ssim1) /2.0 ))
            cyc_ssim_loss2 = tf.cond( RNG2 , lambda:0.0, lambda: -tf.math.log( ( 1.0 + ssim2) /2.0 ))
            cyc_ssim_loss3 = tf.cond( RNG3 , lambda:0.0, lambda: -tf.math.log( ( 1.0 + ssim3) /2.0 ))
            cyc_ssim_loss4 = tf.cond( RNG4 , lambda:0.0, lambda: -tf.math.log( ( 1.0 + ssim4) /2.0 ))
            cyc_ssim_loss5 = tf.cond( RNG5 , lambda:0.0, lambda: -tf.math.log( ( 1.0 + ssim5) /2.0 ))
            self.ssim_cyc_loss  = (cyc_ssim_loss1 + cyc_ssim_loss2 + cyc_ssim_loss3 + cyc_ssim_loss4 + cyc_ssim_loss5 * 10) / 5.0
            # #
            # # ssim_loss_orig = -tf.math.log( (1.0+ssim_recon)/2.0)
            # # ssim_loss = (ssim_loss_orig + ssim_cyc_loss)/6.


            # # SSIM+L1 loss based on Zhao et al (2018) https://arxiv.org/pdf/1511.08861.pdf
            # # Calculated as : self.alpha * loss_MSSSIM + (1-self.alpha) * loss_L1
            # self.MS_SSIML1_loss = self.alpha * self.ssim_cyc_loss + ( 1-self.alpha ) * self.L1_loss_Gen


            # --------------------      SPECULAR       --------------------
            # Specular Loss - To foce ED generation
            Spec_loss1  = tf.reduce_mean( tf.math.square ( (cyc_gen0_yuv   * self.specular_candidate) - (ds1_yuv * self.specular_candidate ) ) )
            Spec_loss2  = tf.reduce_mean( tf.math.square ( (cyc_gen45_yuv  * self.specular_candidate) - (ds2_yuv * self.specular_candidate ) ) )
            Spec_loss3  = tf.reduce_mean( tf.math.square ( (cyc_gen90_yuv  * self.specular_candidate) - (ds3_yuv * self.specular_candidate ) ) )
            Spec_loss4  = tf.reduce_mean( tf.math.square ( (cyc_gen135_yuv * self.specular_candidate) - (ds4_yuv * self.specular_candidate ) ) )
            Spec_lossED = tf.reduce_mean( tf.math.square ( (cyc_genED_yuv  * self.specular_candidate) - (ds5_yuv * self.specular_candidate ) ) )

            ## L2 Norm inpainting loss
            ## Loss = L2_norm(1-Mask x (Igen - Igt)) + L2_norm(Mask x (Igen-Igt))
            # Spec_loss1 = tf.norm( ( 1-self.specular_candidate ) * (cyc_gen0 - ds1_yuv)   , ord=2 ) + tf.norm( self.specular_candidate * (cyc_gen0 - ds1_yuv)   , ord=2 )
            # Spec_loss2 = tf.norm( ( 1-self.specular_candidate ) * (cyc_gen45 - ds2_yuv)  , ord=2 ) + tf.norm( self.specular_candidate * (cyc_gen45 - ds2_yuv)  , ord=2 )
            # Spec_loss3 = tf.norm( ( 1-self.specular_candidate ) * (cyc_gen90 - ds3_yuv)  , ord=2 ) + tf.norm( self.specular_candidate * (cyc_gen90 - ds3_yuv)  , ord=2 )
            # Spec_loss4 = tf.norm( ( 1-self.specular_candidate ) * (cyc_gen135 - ds4_yuv) , ord=2 ) + tf.norm( self.specular_candidate * (cyc_gen135 - ds4_yuv) , ord=2 )
            # Spec_loss5 = tf.norm( ( 1-self.specular_candidate ) * (cyc_genED - ds5_yuv)  , ord=2 ) + tf.norm( self.specular_candidate * (cyc_genED - ds5_yuv)  , ord=2 )

            self.Spec_loss = ( Spec_loss1 + Spec_loss2 + Spec_loss3 + Spec_loss4 ) / 5.0 + ( Spec_lossED ) * 5.0

            # --------------------      StyleTx       --------------------
            # Gram Matrix = [HxW, ch] * [HxW, ch]'
            # calculated after actiavtion of the layer
            # What if we take STYLE = ED and CONTENT = I0 ?

            # CONTENT LOSS:    (Content == I0 == RGB input)
            self.content_loss = tf.reduce_mean( tf.math.square( cyc_genED_yuv - ds1_yuv ) )

            # STYLE LOSS: (Style == ED)
            factor = tf.cast(1/tf.math.square(2 *  9 * self.image_size * self.image_size), tf.float32 )

            gram_STYLE = self.gram_matrix(cyc_genED_yuv)
            gram_CONTENT = self.gram_matrix(ds5_yuv)
            self.style_loss = tf.math.multiply(factor, tf.reduce_mean( tf.math.square( gram_STYLE - gram_CONTENT )))

            # TOTAL LOSS:
            style_weight = 100
            content_weight = 1
            self.total_NST_loss = style_weight * self.style_loss + content_weight * self.content_loss

            # --------------------      TOTAL LOSSES       --------------------
            self.total_Generator_loss      =    ( self.D1_RealFake_loss + self.D3_RealFake_cyc ) / 6.0  + \
                                                ( self.L1_loss_Gen ) * 10.0                             + \
                                                ( self.ssim_cyc_loss ) * 10.0                           + \
                                                ( self.total_NST_loss ) * 10.0
                                                # ( self.Spec_loss ) * 5.0                                + \
                                                # ( self.MS_SSIML1_loss )                                + \


            self.total_Discriminator_loss  =    ( self.D1_classification_loss + self.D3_classification_loss ) / 6.0 + \
                                                ( self.D2_RealFake_target + self.D4_RealFake_cyc ) / 6.0            + \
                                                ( self.D4_classification_loss ) * 0.5                               + \
                                                ( self.total_NST_loss ) * 10
                                                # ( self.Spec_loss ) * 5.0                                            + \


            self.total_Classification_loss =    ( self.D4_classification_loss + self.total_NST_loss) * 10


        """
        # ------------------------------------------------
        #       GRADIENT-TAPE FOR LEARNING PARAMETERS
        # Calculate the gradients for generator and discriminator
        # NOTE: Gradient tape are calculated after the tf.Gradientape() function ends
        # ------------------------------------------------
        """
        # Unfreeze the weights for applying gradients - Important!
        self.G.trainable = True
        self.D.trainable = True

        # https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer
        scaled_gradient_Discriminator = tape.gradient( [  self.total_Discriminator_loss, self.total_Classification_loss ], self.D.trainable_variables)
        scaled_gradient_Discriminator = [(tf.clip_by_value(grad, clip_value_max=1.0, clip_value_min=-1.0)) for grad in scaled_gradient_Discriminator]
        # unscaled_gradient_Discriminator  = self.optimizer_D.get_unscaled_gradients(scaled_gradient_Discriminator)
        self.optimizer_D.apply_gradients( zip( scaled_gradient_Discriminator, self.D.trainable_variables ), experimental_aggregate_gradients=True )
        self.gradmapD  = scaled_gradient_Discriminator

        if self.epoch >= self.train_G_after:
            # self.G.trainable = True
            # self.D.trainable = False
            scaled_gradient_Generator = tape.gradient( [ self.total_Generator_loss], self.G.trainable_variables )
            scaled_gradient_Generator = [(tf.clip_by_value(grad, clip_value_max=1.0, clip_value_min=-1.0)) for grad in scaled_gradient_Generator]
            # unscaled_gradient_Generator = self.optimizer_G.get_unscaled_gradients(scaled_gradient_Generator)
            self.optimizer_G.apply_gradients( zip( scaled_gradient_Generator, self.G.trainable_variables ), experimental_aggregate_gradients=True )
            self.gradmapG  = scaled_gradient_Generator


        return

    """
    # ------------------------------------------------
    #
    # ███    ███  █████  ██ ███    ██
    # ████  ████ ██   ██ ██ ████   ██
    # ██ ████ ██ ███████ ██ ██ ██  ██
    # ██  ██  ██ ██   ██ ██ ██  ██ ██
    # ██      ██ ██   ██ ██ ██   ████
    # ------------------------------------------------
    """
    # jit_compile=True
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def train( self, args ):
        # Set to train on CPU
        # os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        # trainTime = time.time()
        start_time = default_timer()


        file_writer = tf.summary.create_file_writer( self.log_dir )

        tf.Graph.finalize( self._graph )

        # ------------------------------------------------
        # Initialize and load the zipped dataset
        # NOTE: Images returned will be resized, Normalized (TBD) and randomly flipped (TBD)
        self.length_dataset, Dataset = datasetLoad( self )

        # NOTE: Batch size is 1 for zipped dataset, so that 1 image from each polar angle is picked
        # batched_dataset = Dataset.batch( 4 )

        # ------------------------------------------------

        self.G = self.build_generator( )
        self.D = self.build_discriminator( )
        # Print Model summary to console and file
        self.G.summary()
        self.D.summary()
        with open('Generator_summary.txt', 'w') as f:
            self.G.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('Discriminator_summary.txt', 'w') as f:
            self.D.summary(print_fn=lambda x: f.write(x + '\n'))


        # plot the model
        # plot_model( self.D, to_file = 'discriminator_plot.png', show_shapes = True, show_layer_names = True )
        # plot_model( self.G, to_file = 'generator_plot.png', show_shapes = True, show_layer_names = True )

        # self.G.save('G2.h5')

        # ------------------------------------------------
        # Importing the SpecSeg model. The model is initialized with M,N,ch with 3 channels
        self.SpecSeg = SpecSeg(self.image_size, self.image_size, 1)
        self.SpecSeg = tf.keras.models.load_model('specsegv3_chkpt.h5',compile=False)

        self.SpecSeg.summary()
        with open('SpecSeg_summary.txt', 'w') as f:
            self.SpecSeg.summary(print_fn=lambda x: f.write(x + '\n'))
        # ------------------------------------------------
        # Initiialize the checkpoint manager
        checkpoint_dir    = self.checkpoint_save_dir
        ckpt = tf.train.Checkpoint( generator     = self.G,
                                    discriminator = self.D,
                                    optimizer_D   = self.optimizer_D,
                                    optimizer_G   = self.optimizer_G )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        # if old checkpoints exist and flag is true; del them
        if self.delete_old_checkpoints is True and ckpt_manager.latest_checkpoint:
            os.system("echo CLEANUP: Removing previous checkpoints")
            os.system("rm /home/atif/Documents/checkpoints/*")
        elif self.delete_old_checkpoints is False:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

        # ------------------------------------------------
        # Make the iterator from the zipped datasets
        iterator = iter(Dataset)

        batches_per_epoch = int( self.length_dataset/self.batch_size )
        self.batch_step = 0

        plt.close("all")
        gc.collect()

        # self.comet_experiment.set_model_graph(self.D)
        # self.comet_experiment.set_model_graph(self.G)

        # initialize confustion matrix
        # confusion_matrix = self.comet_experiment.create_confusion_matrix(labels=["0°", "45°", "90°", "135°", "ED"])

        for epoch in range(self.num_epochs):

            self.epoch = epoch
            print(f"\nStart of Training Epoch {self.epoch}")
            # since batch will be 1 (to get all polar images in sequence), the number of batches
            # will have to be equal to the length of the dataset. Also note that the randomness and shuffle
            # are not in the dataset while loading, but introduced by randomizing input channels
            # and randomizing lables while training (tf.cond ....)


            for batch in range( batches_per_epoch-1 ):
                self.batch_step +=1

                # Randomly flip all images in the batch
                self.random_flip = tf.random.uniform( [], dtype=tf.float16 ) >= 0.5

                # Randomly generate target labels for more robustness instead of a hard value of 1
                self.TARGET_LABELS = tf.random.uniform( [],  minval=0.8, maxval=1.2, dtype=tf.float32 )


                # Get the next set of images
                element = iterator.get_next()

                orig0   = element[0]
                orig45  = element[1]
                orig90  = element[2]
                orig135 = element[3]
                origED  = element[4]

                self.train_step( orig0, orig45, orig90, orig135, origED )

                with file_writer.as_default():

                    # # # Image grid
                    # figure1 = image_grid( self.cyc_gen0_rgb, self.cyc_gen45_rgb, self.cyc_gen90_rgb, self.cyc_gen135_rgb, self.cyc_genED_rgb )
                    # figure2 = image_grid( orig0, orig45, orig90, orig135, origED )
                    # figure3 = image_grid( self.attention_map1, self.attention_map2, self.attention_map3, self.attention_map4, self.attention_map4 )

                    # plt.close("all")

                    # # ---------- COMET.ML -------------
                    # Generating a confusion matrix for target angle and predicteabels )

                    # Predicted_labels1 = [ tf.squeeze( self.label_orig0_D4 ) ]
                    # Predicted_labels2 = [ tf.squeeze( self.label_orig45_D4 ) ]
                    # Predicted_labels3 = [ tf.squeeze( self.label_orig90_D4 ) ]
                    # Predicted_labels4 = [ tf.squeeze( self.label_orig135_D4 ) ]
                    # Predicted_labels5 = [ tf.squeeze( self.label_origED_D4 ) ]
                    # Target_labels1    = [ tf.squeeze( self.target_label_0deg ) ]
                    # Target_labels2    = [ tf.squeeze( self.target_label_45deg ) ]
                    # Target_labels3    = [ tf.squeeze( self.target_label_90deg ) ]
                    # Target_labels4    = [ tf.squeeze( self.target_label_135deg ) ]
                    # Target_labels5    = [ tf.squeeze( self.target_label_ED ) ]

                    # confusion_matrix.compute_matrix( Target_labels1, Predicted_labels1 )
                    # confusion_matrix.compute_matrix( Target_labels2, Predicted_labels2 )
                    # confusion_matrix.compute_matrix( Target_labels3, Predicted_labels3 )
                    # confusion_matrix.compute_matrix( Target_labels4, Predicted_labels4 )
                    # confusion_matrix.compute_matrix( Target_labels5, Predicted_labels5 )

                    # Log some metrics but only 1 time and not every loop
                    # if self.batch_step == 1:
                        # logging other parameters for record peurposes
                        # self.comet_experiment.log_other( value = self.train_G_after, key="Train G after n loops")
                        # self.comet_experiment.log_other( value = self.TARGET_LABELS, key="Target Accuracy")
                        # self.comet_experiment.log_other( value = self.use_lsgan, key="Use LSGAN")
                        # self.comet_experiment.log_other( value = self.seed, key="Seed Value")
                        # self.comet_experiment.log_other( value = self.randomness, key="Randomness")
                        # self.comet_experiment.log_other( value = self.dropout_amnt, key="Dropout amount")


                    # # Monitoring Losses
                    # log everything every n steps other than confusion matrices. Reduces time per epoch
                    # if self.batch_step % 25 == 0:
                    #     self.comet_experiment.log_metric ( "Total Generator Loss", tf.squeeze(self.total_Generator_loss ), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "G_gan_loss", tf.squeeze(self.G_gan_loss), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "G_L1_loss", tf.squeeze(self.L1_loss_Gen), step=self.batch_step, epoch = self.epoch )

                    #     self.comet_experiment.log_metric ( "Total Discriminator Loss", tf.squeeze(self.total_Discriminator_loss ), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "D_D4_RF_cyc", tf.squeeze(self.D4_RealFake_cyc/6.0), step=self.batch_step, epoch = self.epoch )

                    #     self.comet_experiment.log_metric ( "Total Classification Loss", tf.squeeze(self.total_Classification_loss ), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "D_D4_classf", tf.squeeze(self.D4_classification_loss/5.0), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "G_clsf_loss", tf.squeeze(self.G_clsf_loss), step=self.batch_step, epoch = self.epoch )

                    #     self.comet_experiment.log_metric ( "SPEC LOSS", tf.squeeze(self.Spec_loss), step=self.batch_step, epoch = self.epoch )

                    #     self.comet_experiment.log_metric ( "Content Loss", tf.squeeze(self.content_loss), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "Style Loss", tf.squeeze(self.style_loss), step=self.batch_step, epoch = self.epoch )
                    #     self.comet_experiment.log_metric ( "NST Loss", tf.squeeze(self.total_NST_loss), step=self.batch_step, epoch = self.epoch )

                        # # close plots to clear up memory. Not required if using plot_to_image() ftn
                        # logs progperly but causes memory leakage.
                        # self.comet_experiment.log_figure( figure=debug_plot(self.gen_input), figure_name="Inp * mask", step=tensorboard_step)
                        # self.comet_experiment.log_figure( figure=figure3, figure_name="attention", step=self.batch_step)

                        # Plotting output of G1
                        # self.comet_experiment.log_image( tf.squeeze( (self.gen_Y) ), name="G1 Y-ch",         step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze((self.gen_rgb_output)), name="G1 RGB",     step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze(self.target_img), name="1. Target Gen ", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze(origED), name="2. ED Image ", step=self.batch_step)
                        # # self.comet_experiment.log_image( tf.squeeze( tf.math.sigmoid(self.gen_Y)*255 ), name="G1 (Scaled)",         step=self.batch_step)

                        # # plotting all cyclic images to compare if Generator is learning
                        # self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen0_rgb) ), name="G2 0°", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen45_rgb) ), name="G2 45°", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen90_rgb) ), name="G2 90°", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen135_rgb) ), name="G2 135°", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze( (self.cyc_genED_rgb) ), name="G2 ED", step=self.batch_step)

                        # self.comet_experiment.log_image( tf.squeeze(self.specular_candidate), name="Specular Mask ", step=self.batch_step)
                        # self.comet_experiment.log_image( tf.squeeze(self.gen_rgb_normalized),       name="G1 RGB Norm",     step=self.batch_step)

                        # plt.close("all")
                        # gc.collect()

                    # Onlt log histogram every n steps, to avoid comet upload rate issue
                    # alos doesnt slow down training
                    if self.batch_step % 100 == 0:
                        for index, grad1 in enumerate(self.gradmapD):
                            self.comet_experiment.log_histogram_3d( self.gradmapD[index], name="Discriminator Grads", step=self.batch_step, epoch=self.epoch )
                        for index2, grad2 in enumerate(self.gradmapG):
                            self.comet_experiment.log_histogram_3d( self.gradmapG[index2], name="Generator Grads", step=self.batch_step, epoch=self.epoch)

                    # CLEAR UP MEMORY
                    file_writer.flush()
                    plt.close("all")
                    gc.collect()

                    # print the progress (taken from stargan-github)
                    printProgressBar( (batch+1) % 1000, batches_per_epoch-1, decimals=2)

             #  Print losses every x epochs
            if (self.epoch + 1) % self.log_step == 0:
                finish = default_timer()
                print( "\n")
                # print( f"\tIteration: [{self.epoch + 1}/{self.log_step}]" )
                print ('Time taken for epoch {} is {} min\n'.format(self.epoch + 1, (finish-start_time)/60))

                # Plot Confusion matrix for every epoch
                # self.comet_experiment.log_confusion_matrix( matrix = confusion_matrix,
                #                                             title="Confusion Matrix D2, Epoch #%d" % (self.epoch + 1),
                #                                             file_name="confusion-matrix-%03d.json" % (self.epoch + 1),
                #                                             labels=["0°", "45°", "90°", "135°", "ED"])


                # In eager mode, grads does not have name, so we get names from model.trainable_weights
                # for weights, grads in zip(self.D.trainable_weights, self.gradmapD):
                #     tf.summary.histogram(weights.name.replace(':', '_')+'_grads', data=grads, step=self.batch_step)

                # gc.collect()

                # Stop the trace and export the collected information
                # tf.summary.trace_export(name="Train", step=self.batch_step, profiler_outdir=self.log_dir)

            # Save the weights at every x steps
            if (self.epoch + 1) % self.checkpoint_save_step == 0:
                # save checkpoint -
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(self.epoch+1, ckpt_save_path))

        print( f"Time for training was {(time.time() - start_time) / 60.0} minutes" )

        # Save last checkpoint before quitting
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(self.epoch+1, ckpt_save_path))
        file_writer.flush()
        gc.collect()
        # close tensorboard writer
        file_writer.close()
        return

    # ------------------------------------------
    #
    # ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
    # ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
    # █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
    # ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
    # ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████
    #
    # ------------------------------------------


    # ------------------------------------------
    # To calcualte the Degree of Polarization from the polarimetric images
    # Input: 4 polarimetric image Y channel
    # Output: Image with values equal to Degree of polarization
    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def calcDOP( self, I0_Ych, I45_Ych, I90_Ych, I135_Ych ):
        S0 =  I0_Ych +  I90_Ych
        S1 =  I0_Ych -  I90_Ych
        S2 = I45_Ych - I135_Ych

        PolIntensity = tf.math.sqrt( tf.math.square(S1) + tf.math.square(S2) )
        DoP = tf.math.divide_no_nan( PolIntensity, S0)
        0.5 * tf.math.atan2( S2, S1 )

        # test
        plot_single_image(DoP)  # imported from Utils

        return DoP

    # ------------------------------------------
    # Calculating Gram Matrix
    # Sauce: https://www.tensorflow.org/tutorials/generative/style_transfer
    # Input: Single image tensor
    # Output: gram matrix value
    def gram_matrix( self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)


    # ------------------------------------------
    # Calculating Gram Matrix
    # Sauce: https://www.datacamp.com/community/tutorials/implementing-neural-style-transfer-using-tensorflow
    # Input: Single image tensor
    # Output: gram matrix value
    def gram_matrix2( self, input_tensor):
        temp = input_tensor
        temp = tf.squeeze(temp)
        tf.reshape(temp, [temp.shape[2], temp.shape[0]*temp.shape[1]])
        result = tf.matmul(temp, temp, transpose_b=True)
        gram = tf.expand_dims(result, axis=0)
        return gram

    # ------------------------------------------
    #                   FID
    # ------------------------------------------
    # Calculating Frechet-Inception-Distance (FID)
    # Sauce: https://github.com/Atif-Anwer/PolarCycle_Anon/blob/master/scripts/evaluate_fid.py
    # Input: Two images to compare
    # Output: FID Value
    def tf_cov( self, x):
        mean_x = tf.reduce_mean(x)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx

    def tf_sqrtm_sym( self, mat, eps=1e-10):
        # WARNING : This only works for symmetric matrices !
        s, u, v = tf.svd(mat)
        si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
        return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)

    def calculate_FID( self, image1, image2):
        mu_real = tf.reduce_mean(image1)
        mu_fake = tf.reduce_mean(image2)

        sigma_real = self.tf_cov(image1)
        sigma_fake = self.tf_cov(image2)

        diff = mu_real - mu_fake
        mu2 = tf.reduce_sum(tf.multiply(diff, diff))

        # Computing the sqrt of sigma_real * sigma_fake
        sqrt_sigma = self.tf_sqrtm_sym(sigma_real)
        # sqrt_sigma = tf.transpose(sigma_fake)
        sqrt_a_sigma_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_fake, sqrt_sigma))

        tr = tf.trace(sigma_real + sigma_fake - 2 * self.tf_sqrtm_sym(sqrt_a_sigma_a))
        fid = mu2 + tr
        return fid


    # ------------------------------------------
    #              INCEPTION SCORE
    # ------------------------------------------
    # Sauce: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
    # def calculate_inception_score(images, n_split=10, eps=1E-16):
    #     n_part = floor(images.shape[0] / n_split)
    #     for i in range(n_split):
    #         # retrieve p(y|x)
    #         ix_start, ix_end = i * n_part, i * n_part + n_part
    #         p_yx = yhat[ix_start:ix_end]
    #         # calculate p(y)
    #         p_y = expand_dims(p_yx.mean(axis=0), 0)
    #         # calculate KL divergence using log probabilities
    #         kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    #         # sum over classes
    #         sum_kl_d = kl_d.sum(axis=1)
    #         # average over images
    #         avg_kl_d = mean(sum_kl_d)
    #         # undo the log
    #         is_score = exp(avg_kl_d)
    #         # store
    #         scores.append(is_score)
    #     # average across images
    #     is_avg, is_std = mean(scores), std(scores)
    #     return is_avg, is_std

        # # pretend to load images
        # images = ones((50, 299, 299, 3))
        # print('loaded', images.shape)
        # # calculate inception score
        # is_avg, is_std = calculate_inception_score(images)
        # print('score', is_avg, is_std)


    @tf.function(experimental_follow_type_hints=True, jit_compile=True)
    def custom_per_image_standardization(self, image):

        """
        Custom function taken from TF's original per image standardiztization function
        Returns image tensor with mean , norm=1
        """
        # image = ops.convert_to_tensor(image, name='image')

        # num_pixels = math_ops.reduce_prod(array_ops.shape(image))
        num_pixels = tf.constant( 65536, dtype=tf.float32 )  # for 256x256

        image = tf.cast(image, dtype=tf.float32)
        # image_mean = math_ops.reduce_mean(image)
        image_mean = tf.math.reduce_mean(image)

        # variance = (math_ops.reduce_mean(math_ops.square(image)) - math_ops.square(image_mean))
        variance = ( tf.math.reduce_mean( tf.math.square( image ) ) ) - tf.math.square (image_mean)
        # variance = gen_nn_ops.relu(variance)
        variance = tf.nn.relu( variance )
        # stddev = math_ops.sqrt(variance)
        stddev = tf.cast ( tf.math.sqrt( variance ), tf.float32 )

        # Apply a minimum normalization that protects us against uniform images.
        # min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
        min_stddev = tf.math.rsqrt ( tf.cast (num_pixels, tf.float32)  )
        # pixel_value_scale = math_ops.maximum(stddev, min_stddev)
        pixel_value_scale = tf.math.maximum( stddev, min_stddev )
        # pixel_value_scale = stddev
        pixel_value_offset = image_mean

        # image = tf.math.subtract(image, pixel_value_offset)
        image = tf.math.divide(image, pixel_value_scale)

        # append the calculated values for restoring images
        self.variance_arr.append( variance )
        self.stddev_arr.append( pixel_value_scale )
        self.mean_arr.append( pixel_value_offset )

        return image
    # ------------------------------------------------
    # REFERENCES:
    # ------------------------------------------------
    # CYCLEGAN:             https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=KBKUV2sKXDbY
    # CYCLEGAN:             https://www.tensorflow.org/tutorials/generative/cyclegan
    # STARGAN:              https://github.com/Kal213/StarGAN-Tutorial-Tensorflow-2.3/blob/main/StarGAN.py
    # PIX2PIX:              https://www.tensorflow.org/tutorials/generative/pix2pix
    # BCE:                  https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
    # LSGAN:                https://jonathan-hui.medium.com/gan-lsgan-how-to-be-a-good-helper-62ff52dd3578
    # SIGMOID SOFTMAX:      https://medium.com/arteos-ai/the-differences-between-sigmoid-and-softmax-activation-function-12adee8cf322
    # GRADIENT CHECKING:    https://www.youtube.com/watch?v=P6EtCVrvYPU
    # GAN LOSS:             https://neptune.ai/blog/gan-loss-functions
    # GAN LOSS:             https://developers.google.com/machine-learning/gan/loss
    # IMAGE COMPLETION:     https://bamos.github.io/2016/08/09/deep-completion/
    # OVERFITTING:          https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/
    # PATCHGAN:             https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
    # BATCH NORMALIZATION:  https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338
    # VISUALIZE FILTERS:    https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    # VISUALIZE FILTERS:    https://keisen.github.io/tf-keras-vis-docs/examples/visualize_conv_filters.html
    # https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21
    # https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
    # https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
    # FORMULAE: https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967
    # RESIZING ISSUES: https://pythonrepo.com/repo/GaParmar-clean-fid-python-deep-learning

    # SAGAN: https://github.com/leafinity/SAGAN-tensorflow2.0/blob/b09b035ce39699b724a212e260ce3cf2f03760c1/attention.py#L4


    # Ubuntu Performance
    # https://askubuntu.com/questions/604720/setting-to-high-performance
    # MIXED PRECISION: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
    # MIXED PRECISION: https://www.tensorflow.org/guide/mixed_precision
