import gc
import pathlib
import pickle
import time

# from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from comet_ml import Experiment
from matplotlib import pyplot as plt
from skimage.color import deltaE_cie76, deltaE_ciede94
from tabulate import tabulate

# Import SpecSeg network
from SpecSeg import SpecSeg
from utils import (
        rescale_01,
)

"""
# ------------------------------------------------
#
# ████████ ███████ ███████ ████████
#    ██    ██      ██         ██
#    ██    █████   ███████    ██
#    ██    ██           ██    ██
#    ██    ███████ ███████    ██
#
# TEST FUNCTION FOR SHMGAN
# The test function has the following features:
# - Load RGB image from test folder as I0 (or Itot)
# - Set all other layers to zero
# - Set target image label as ED
# - Average CbCr is replaced with CbCr of the image
# - Generate images. Both G1 and G_cyclic
# - No need for losses
# ------------------------------------------------
    """
def test( self, args ):

        self.comet_experiment = Experiment(
                api_key                            = "insert your comet api key here",
                project_name                       = "insert your project name here",
                workspace                          = "insert your workspace here",
                auto_param_logging                 = True,
                auto_metric_logging                = True,
                log_env_details                    = True,
                log_code                           = True,                        # code logging
                log_graph                          = True,
                log_env_gpu                        = True,
                log_env_host                       = True,
                log_env_cpu                        = True,
                auto_histogram_tensorboard_logging = True,
                auto_histogram_weight_logging      = True,
                auto_histogram_gradient_logging    = True,
                auto_histogram_activation_logging  = True,
                # auto_histogram_epoch_rate=1,
            )
        self.comet_experiment.add_tag("TEST RUN")

        # Do not flip the image
        self.random_flip = 0.0
        # do not randomize target label values
        self.TARGET_LABELS = 1.0

        # Disable deleting by mistake
        self.delete_old_checkpoints = False

        # Step1: Load the test images
        rootfolder = args.test_dir
        testpath = pathlib.Path( rootfolder )
        # NOTE: While loading the images, only difference is that the images are not flipped. Otherwise it is the same function as
        # the dataset loading images

        # Intialize array for saving each image's values
        self.stddev_arr   = []
        self.mean_arr     = []
        self.variance_arr = []

        test_images = tf.keras.preprocessing.image_dataset_from_directory(
                str( testpath ),
                  labels           = None,
                # label_mode       = 'categorical',
                  color_mode       = 'rgb',
                  validation_split = None,
                  shuffle          = False,
                  seed             = 1337,
                  image_size       = (self.image_size, self.image_size),
                  batch_size       = 1
                ) \
                .cache() \
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE ) \
                .prefetch(25)
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE ) \
        test_images.class_names = 'TEST'

        # Only load diffuse images if the flag is True
        if args.calc_metrics is True:
            diffusefolder = args.diffuse_dir
            diffusepath = pathlib.Path( diffusefolder )
            # NOTE: While loading the images, only difference is that the images are not flipped. Otherwise it is the same function as
            # the dataset loading images
            diffuse_images = tf.keras.preprocessing.image_dataset_from_directory(
                    str( diffusepath ),
                    labels           = None,
                    # label_mode       = 'categorical',
                    color_mode       = 'rgb',
                    validation_split = None,
                    shuffle          = False,
                    seed             = 1337,
                    image_size       = (self.image_size, self.image_size),
                    batch_size       = 1
                    ) \
                    .cache() \
                    .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE ) \
                    .prefetch(25)
                    # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                    # .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                    # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE ) \
            test_images.class_names = 'TEST'

        # return the number of files loaded
        self.number_of_test_images = len(np.concatenate([i for i in test_images], axis=0))

        # ZIP the datasets into one dataset
        if args.calc_metrics is True:
            loadedDataset = tf.data.Dataset.zip ( ( test_images, diffuse_images ) )
        else:
            loadedDataset = tf.data.Dataset.zip ( test_images )

        options = tf.data.Options()
        options.threading.max_intra_op_parallelism = 1
        loadedDataset = loadedDataset.with_options(options)
        loadedDataset = loadedDataset.cache().prefetch( buffer_size =25)

        # Load the G and D
        self.G = self.build_generator( )
        self.D = self.build_discriminator( )
        # Print Model summary to console and file
        self.G.summary()
        self.D.summary()
        with open('Generator_summary.txt', 'w') as f:
            self.G.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('Discriminator_summary.txt', 'w') as f:
            self.D.summary(print_fn=lambda x: f.write(x + '\n'))

        """
        ----------------------------------------------
        Adding SpecSeg to Test
        ----------------------------------------------
        """
        self.SpecSeg = SpecSeg(self.image_size, self.image_size, 1)
        self.SpecSeg = tf.keras.models.load_model('specsegv3_chkpt.h5',compile=False)
        self.SpecSeg.summary()
        with open('SpecSeg_summary.txt', 'w') as f:
            self.SpecSeg.summary(print_fn=lambda x: f.write(x + '\n'))

        # ------------------------------------------------
        # STEP2: Load checkpoints
        checkpoint_dir    = self.checkpoint_save_dir
        ckpt = tf.train.Checkpoint( generator     = self.G,
                                    discriminator = self.D,
                                    optimizer_D   = self.optimizer_D,
                                    optimizer_G   = self.optimizer_G )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print ('Latest checkpoint restored!!')

        plt.close("all")
        gc.collect()

        # STEP3: Iterate over the loaded test images
        test_iterator = iter(loadedDataset)

        # create zeros and ones for labels
        tmp_zeros = tf.zeros( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )
        trg_ones  = tf.ones( [1, self.image_size, self.image_size, 1], dtype=tf.float32 )

        # initialize lists for printing data
        MSE   = []
        SSIM  = []
        PSNR  = []
        index = []
        table = []
        delE76 = []
        delE94 = []
        processing_time_taken = []

        print('\n\n\n->> "I\'m sorry, Dave. You will have to wait a little while I process... Regards, HAL 9000 ◍ <<- \n\n\n')

        # for all images in the test folder
        for i in range(self.number_of_test_images):

            self.start_time = time.time()

            # Randomly generate target labels for more robustness instead of a hard value of 1
            self.TARGET_LABELS = tf.random.uniform( [],  minval=0.8, maxval=1.2, dtype=tf.float32 )

            # Get the image
            element = test_iterator.get_next()
            if args.calc_metrics is True:
                self.rgb_testImage   = element[0]
                self.rgb_diffuseImage  = element[1]
            else:
                self.rgb_testImage   = element

            # Setting target labels for Cyclic generation
            self.target_label_ED     = tf.Variable( [0,0,0,0,self.TARGET_LABELS],  dtype=tf.float32 )

            # setting both G and D as non-trainable
            self.G.trainable = False
            self.D.trainable = False

            # setting ED as input image and other channels as zero
            RGBInput = self.custom_per_image_standardization( tf.image.rgb_to_yuv( self.rgb_testImage[:, :, :, :] ) )

            # Generating the specular mask from the input RGB image
            self.specular_candidate = (self.SpecSeg.predict(RGBInput[:, :, :, 0, tf.newaxis], verbose=0))

            # setting the CbCr same as the input image
            averageCbCr = RGBInput[:, :, :, 1:]

            # Y channel input are set to zero and the input is 0 degree
            ych_inp1 = RGBInput[:, :, :, 0, tf.newaxis]
            ych_inp2 = tmp_zeros
            ych_inp3 = tmp_zeros
            ych_inp4 = tmp_zeros
            ych_inp5 = tmp_zeros
            # generate the inputs
            rand_input_Ych = tf.concat( [ych_inp1, ych_inp2, ych_inp3, ych_inp4, ych_inp5], axis = 3 )

            self.gen_input          = tf.concat( [rand_input_Ych, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones], axis = 3 )
            self.target_img         = self.rgb_testImage
            self.Target_angle_label = self.target_label_ED

            # test plot the input
            # debug_plot( self.gen_input )

            """--------------------G(1)-------------------"""
            self.gen_Y   = self.G ( self.gen_input, training=False )
            self.gen_YCbCr   = tf.concat( [self.gen_Y, averageCbCr], axis = 3 )

            avg_stddev_arr      = tf.reduce_mean(self.stddev_arr)
            tf.reduce_mean(self.mean_arr)
            # avg_variance_arr    = tf.reduce_mean(self.variance_arr)
            self.gen_rgb_output = tf.image.yuv_to_rgb( ( tf.multiply(self.gen_YCbCr, avg_stddev_arr) ) * 255.0 )
            self.gen_rgb   = tf.image.yuv_to_rgb( self.gen_YCbCr )

            orig_Ych = self.gen_rgb[:, :, :, 0, tf.newaxis]

            # plot_single_image ( self.gen_Y )
            # plot_single_image ( self.gen_rgb, title="Generated RGB" )

            """--------------------G(2)-------------------"""
            # -----------------CONFIG B -------------------
            # NOTE: This gives **slightly** better results than CONFIG-A above
            cyc_Y1 = tf.concat( [tmp_zeros, orig_Ych, orig_Ych, orig_Ych, orig_Ych], axis = 3 )
            cyc_Y2 = tf.concat( [orig_Ych, tmp_zeros, orig_Ych, orig_Ych, orig_Ych], axis = 3 )
            cyc_Y3 = tf.concat( [orig_Ych, orig_Ych, tmp_zeros, orig_Ych, orig_Ych], axis = 3 )
            cyc_Y4 = tf.concat( [orig_Ych, orig_Ych, orig_Ych, tmp_zeros, orig_Ych], axis = 3 )
            cyc_Y5 = tf.concat( [orig_Ych, orig_Ych, orig_Ych, orig_Ych, tmp_zeros], axis = 3 )


            # ####################################################################################
            # Append one-hot tensors as labels
            # NOTE: Use ZEROS as input; same as training.
            # -----------------CONFIG A -------------------
            cyclic_input1 = tf.concat( [cyc_Y1, trg_ones, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input2 = tf.concat( [cyc_Y2, tmp_zeros, trg_ones, tmp_zeros, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input3 = tf.concat( [cyc_Y3, tmp_zeros, tmp_zeros, trg_ones, tmp_zeros, tmp_zeros], axis = 3 )
            cyclic_input4 = tf.concat( [cyc_Y4, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones, tmp_zeros], axis = 3 )
            cyclic_input5 = tf.concat( [cyc_Y5, tmp_zeros, tmp_zeros, tmp_zeros, tmp_zeros, trg_ones], axis = 3 )

            # debug_plot( cyclic_input5 )

            # Generate 5 Y channel CYCLIC images, to reconstruct the original images.
            cyc_0_Y   = self.G( cyclic_input1, training=False )
            cyc_45_Y  = self.G( cyclic_input2, training=False )
            cyc_90_Y  = self.G( cyclic_input3, training=False )
            cyc_135_Y = self.G( cyclic_input4, training=False )
            cyc_ED_Y  = self.G( cyclic_input5, training=False )

            cyc_gen0   = tf.concat( [ cyc_0_Y, averageCbCr], axis = 3 )
            cyc_gen45  = tf.concat( [ cyc_45_Y, averageCbCr], axis = 3 )
            cyc_gen90  = tf.concat( [ cyc_90_Y, averageCbCr], axis = 3 )
            cyc_gen135 = tf.concat( [ cyc_135_Y, averageCbCr], axis = 3 )
            cyc_genED  = tf.concat( [ cyc_ED_Y, averageCbCr], axis = 3 )

            # convert to RGB
            self.cyc_gen0_rgb   = tf.image.yuv_to_rgb( cyc_gen0 )
            self.cyc_gen45_rgb  = tf.image.yuv_to_rgb( cyc_gen45 )
            self.cyc_gen90_rgb  = tf.image.yuv_to_rgb( cyc_gen90 )
            self.cyc_gen135_rgb = tf.image.yuv_to_rgb( cyc_gen135 )
            self.cyc_genED_rgb  = tf.image.yuv_to_rgb( cyc_genED )

            processing_time_taken.append( (time.time() - self.start_time) )

            # self.test_plot()

            #  ---------------------- COMET Logging ------------------------
            # Plotting output of G1
            self.comet_experiment.log_image( tf.squeeze( (self.gen_Y) ), name="G1 Y-ch", step=i)
            self.comet_experiment.log_image( tf.squeeze((self.gen_rgb)), name="G1 RGB", step=i)
            self.comet_experiment.log_image( tf.squeeze(self.target_img), name="1. Input Image ", step=i)

            self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen0_rgb) ), name="G2 0°", step=i)
            self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen45_rgb) ), name="G2 45°", step=i)
            self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen90_rgb) ), name="G2 90°", step=i)
            self.comet_experiment.log_image( tf.squeeze( (self.cyc_gen135_rgb) ), name="G2 135°", step=i)
            self.comet_experiment.log_image( tf.squeeze( (self.cyc_genED_rgb) ), name="G2 ED", step=i)

            self.comet_experiment.log_image( tf.squeeze(self.specular_candidate), name="Specular Mask ", step=i)
            if args.calc_metrics is True:
                self.comet_experiment.log_image( tf.squeeze(self.rgb_diffuseImage), name="2. Target Diffuse ", step=i)
            #  ---------------------- PyPlt printing ------------------------

            # image_grid( self.cyc_gen0_rgb, self.cyc_gen45_rgb, self.cyc_gen90_rgb, self.cyc_gen135_rgb, self.cyc_genED_rgb )
            # plot the generated images :fingerscrossed:
            # plot_single_image ( self.cyc_gen0_rgb )
            # plot_single_image ( self.cyc_gen45_rgb )
            # plot_single_image ( self.cyc_gen90_rgb )
            # plot_single_image ( self.cyc_gen135_rgb )
            # plot_single_image ( self.cyc_genED_rgb )
            # plt.close("all")
            # gc.collect()

            # ----------------- calculating Metrics -------------------
            # calculate only if the flag is true
            if args.calc_metrics is True:
                index.append(i+1)

                # FID_score   = self.calculate_FID( self.cyc_genED_rgb , self.target_img )
                SSIM.append( (tf.image.ssim ( rescale_01( self.gen_rgb ), rescale_01( self.rgb_diffuseImage ), 5 )).numpy() )

                im1 = tf.image.convert_image_dtype(self.gen_rgb, tf.float32)
                im2 = tf.image.convert_image_dtype(self.rgb_diffuseImage, tf.float32)
                # psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
                # PSNR.append( (tf.image.psnr ( tf.clip_by_value(self.gen_rgb, 0, 255), tf.clip_by_value(self.rgb_diffuseImage, 0, 255), max_val=255 )).numpy() )
                PSNR.append( (tf.image.psnr(im1 , im2 , max_val=1.0)).numpy() )

                # Calculate L1 loss to original image?
                # Or use builtin functions to evaluate the Generator?
                L2_loss = tf.keras.losses.MeanSquaredError()
                MSE.append( L2_loss(self.gen_rgb, self.rgb_diffuseImage ).numpy() )

                # print ( 'Processing Image# {}: {:.3f} secs, MSE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f} \n' .format( i, processing_time_taken[i], MSE[i], SSIM[i], PSNR[i]) )

                img1 = tfio.experimental.color.rgb_to_lab( tf.image.convert_image_dtype(self.gen_rgb, dtype=tf.float32) )
                img2 = tfio.experimental.color.rgb_to_lab( tf.image.convert_image_dtype(self.rgb_diffuseImage, dtype=tf.float32) )
                delE76.append ( tf.reduce_mean(tf.py_function(deltaE_cie76, inp=[img1,img2], Tout=tf.float32)) )
                delE94.append ( tf.reduce_mean(tf.py_function(deltaE_ciede94, inp=[img1,img2], Tout=tf.float32)) )

                # populate table
                column = [index[i], processing_time_taken[i], MSE[i], SSIM[i], PSNR[i], delE76[i], delE94[i]]
                table.append(column)

                # # print table inline
                # print( tabulate( table, tablefmt="plain" ))

                self.comet_experiment.log_metric ( "Processing Time", processing_time_taken[i], step=i  )
                self.comet_experiment.log_metric ( "MSE", MSE[i], step=i  )
                self.comet_experiment.log_metric ( "SSIM", SSIM[i], step=i  )
                self.comet_experiment.log_metric ( "PSNR", PSNR[i], step=i  )

        # Print metrics only if flag is true
        if args.calc_metrics is True:
            print('\n\n --- PRINTING ALL CALCUATED METRICS --- ')
            print(tabulate(table, headers=['Image#', 'Time', 'MSE', 'SSIM', 'PSNR', 'delE76', 'delE94']))

            # Calculating mean values
            mean_mse  = sum(MSE) / len(MSE)
            mean_ssim = sum(SSIM) / len(SSIM)
            mean_psnr = sum(PSNR) / len(PSNR)
            mean_delE76 = sum(delE76) / len(delE76)
            mean_delE94 = sum(delE94) / len(delE94)
            print('\n\n --- PRINTING MEAN METRICS --- ')
            mean_metrics = [mean_mse, mean_ssim, mean_psnr, mean_delE76, mean_delE94]
            print(tabulate([mean_metrics], headers=['Mean MSE', 'Mean SSIM', 'Mean PSNR', 'Mean dleE76', 'Mean delE94']))
            print('\n\n' )

            # saving all the calculated metrics as txt
            with open("SSIM.txt", 'wb+') as file1:
                pickle.dump(SSIM, file1)

            with open("MSE.txt", 'wb+') as file2:
                pickle.dump(MSE, file2)

            with open("PSNR.txt", 'wb+') as file3:
                pickle.dump(PSNR, file3)

            # logging means to Comet also before closing experiment
            # self.comet_experiment.log_other( value = MSE, key="All MSE")
            # self.comet_experiment.log_other( value = SSIM, key="All SSIM")
            # self.comet_experiment.log_other( value = PSNR, key="All PSNR")
        #     self.comet_experiment.log_other( value = mean_mse,  key="Mean MSE")
        #     self.comet_experiment.log_other( value = mean_ssim, key="Mean SSIM")
        #     self.comet_experiment.log_other( value = mean_psnr, key="Mean PSNR")

        self.comet_experiment.end()

        print('\n\n\n->> "Thank you for a very enjoyable game - HAL 9000 ◍ <<- \n\n\n')

        return

# ------------------------------------------
# PLOTTING TEST IMAGES POST TRAINING
def test_plot( self ):
        figure = plt.figure( figsize=(10,15) )
        figure.add_subplot( 2, 1, 1, title="Orig")
        plt.imshow ( tf.squeeze( rescale_01( self.rgb_testImage ) ).numpy().astype("float32") )
        figure.add_subplot( 2, 1, 2, title="Generated G1")
        plt.imshow ( tf.squeeze( rescale_01( self.gen_rgb ) ).numpy().astype("float32") )

        figure = plt.figure( figsize=(10,15) )
        figure.add_subplot( 2, 2, 1, title="Cyc0")
        plt.imshow ( tf.squeeze( rescale_01( self.cyc_gen0_rgb ) ).numpy().astype("float32") )
        figure.add_subplot( 2, 2, 2, title="Cyc45")
        plt.imshow ( tf.squeeze( rescale_01( self.cyc_gen45_rgb ) ).numpy().astype("float32") )
        figure.add_subplot( 2, 2, 3, title="Cyc90")
        plt.imshow ( tf.squeeze( rescale_01( self.cyc_gen90_rgb ) ).numpy().astype("float32") )
        figure.add_subplot( 2, 2, 4, title="Cyc135")
        plt.imshow ( tf.squeeze( rescale_01( self.cyc_gen135_rgb ) ).numpy().astype("float32") )
