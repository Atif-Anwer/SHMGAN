"""
# -----------------------------------------------------------
# SHMGAN -  Removal of Specular Highlights by a GAN Network
#

# Uses Packages:
#     Python 3.8
#     CUDA 11.8
#     cuDnn 8.0
#     Tensorflow 2.8

# (C) 2023 Atif Anwer, INSA Rouen, France
# Email: atif.anwer@insa-rouen.fr
# -----------------------------------------------------------
"""
import argparse
import os
from distutils.util import strtobool
from test import test

import tensorflow as tf

# import comet_ml at the top of your file
from packaging import version

from ShmGANwithSSpecSeg import ShmGANwithSSpecSeg
from utils import check_gpu


def parse_args():

    desc = "SHMGAN for specular highlight mitigation"
    parser = argparse.ArgumentParser( description = desc )

    # Flags
    parser.add_argument( '--est_diffuse', type = bool, default = True, help = '(TRUE) Estimate diffuse image from images or (FALSE) load from hdf5 file' )
    parser.add_argument( '--flip', type = bool, default = True, help = '(TRUE) Flip images randomly while loading dataset' )
    parser.add_argument( '--mode', type = str, default = 'train', choices = ['train', 'test'] )
    parser.add_argument( '--calc_metrics', dest='calc_metrics', default=False, type=lambda x:bool(strtobool(x)), help = '(False) Calculate metrics (PSNR, MSE, SSIM etc)' )
    parser.add_argument( '--delete_old_checkpoints', type = bool, default = True, help = '(True) Delete old checkpoints)' )

    parser.add_argument( '--image_size', type = int, default = 128, help = 'image resize resolution' )
    parser.add_argument( '--batch_size', type = int, default = 1, help = 'mini-batch size' )
    parser.add_argument( '--num_epochs', type = int, default = 200, help = 'Number of epochs' )
    parser.add_argument( '--n_critic', type = int, default = 5, help = 'number of D updates per each G update' )
    parser.add_argument( '--log_step', type = int, default = 1, help = 'Log every x step' )
    parser.add_argument( '--checkpoint_save_step', type = int, default = 10 )

    # Model parameters
    parser.add_argument( '--filter_size', type = int, default = 64, help = 'Initial Filter size for convolution' )
    parser.add_argument( '--c_dim', type = int, default = 5, help = 'dimension of polarimetric domain images )' )
    parser.add_argument( '--g_lr', type = float, default = 0.00002, help = 'learning rate for G' )
    parser.add_argument( '--d_lr', type = float, default = 0.00002, help = 'learning rate for D' )
    parser.add_argument( '--beta1', type = float, default = 0.5, help = 'beta1 for Adam optimizer' )
    parser.add_argument( '--beta2', type = float, default = 0.99, help = 'beta2 for Adam optimizer' )
    parser.add_argument( '--num_iteration_decay', type = int, default = 100000, help = 'number of iterations for decaying lr' )
    parser.add_argument( '--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Directories.
    parser.add_argument( '--data_dir', default = '/media/atif/Expansion/Specular Datasets/PSD_Dataset/PSDpolarForSHMGAN', help = 'Path to polarimetric images' )
    parser.add_argument( '--test_dir', default = '/media/atif/Expansion/Specular Datasets/PSD_Dataset/PSD_Test/PSD_Test_specular', help = 'Path to polarimetric images' )
    parser.add_argument( '--diffuse_dir', default = '/media/atif/Expansion/Specular Datasets/PSD_Dataset/PSD_Test/PSD_Test_diffuse_resized', help = 'Path to diffuse images' )
    parser.add_argument( '--model_save_dir', type = str, default = './models' )
    parser.add_argument( '--checkpoint_save_dir', type = str, default = '/home/atif/Documents/checkpoints' )
    parser.add_argument( '--result_dir', type = str, default = './results' )
    parser.add_argument( '--log_dir', type = str, default = './logs/train' )

    # Step size.
    parser.add_argument( '--num_iteration', type = int, default = 20000, help = 'number of total iterations for training D' )
    return parser.parse_args()

def main():
    # Parse ags
    args = parse_args()
    if len( vars (args) ) < 1:
        # check minimum arguments provided
        print(":facepalm: => Usage : main.py -data_dir etc etc ")
        exit(1)

    # Set to train on GPU
    check_gpu()

    print("TensorFlow version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This notebook requires TensorFlow 2.0 or above."


    q = vars( args )
    print( '------------ Options -------------' )
    for k, v in sorted( q.items() ):
        print( '%s: %s' % (str( k ), str( v )) )
    print( '-------------- End ----------------' )

    # Delete Previous Tensflow Logs
    os.system("echo -------------------------------")
    os.system("echo CLEANUP: Removing previous logs")
    os.system("rm ./logs/train/*")
    # os.system("rm ./logs/*")
    os.system("echo -------------------------------")

    # setup model
    # Class includes loading dataset
    shmgan = ShmGANwithSSpecSeg( args )

    # train or test, as required
    if args.mode == 'train':
        shmgan.train( args )
        print( " [*] Training finished!" )
    elif args.mode == 'test':
        test( shmgan, args )



# ------------------------------------------------
if __name__ == "__main__":
    # Reduces Tensorflow messages other than errors or important messages
    # '0' #default value, output all information
    # '1' #Block notification information (INFO)
    # '2' #Shield notification information and warning information (INFO\WARNING)
    # '3' #Block notification messages, warning messages and error messages (INFO\WARNING\FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
