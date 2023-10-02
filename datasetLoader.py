import os
import pathlib

import numpy as np
import tensorflow as tf


# ------------------------------------------------
#
# ██████   █████  ████████  █████  ██       ██████   █████  ██████  ███████ ██████
# ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████    ██    ███████ ██      ██    ██ ███████ ██   ██ █████   ██████
# ██   ██ ██   ██    ██    ██   ██ ██      ██    ██ ██   ██ ██   ██ ██      ██   ██
# ██████  ██   ██    ██    ██   ██ ███████  ██████  ██   ██ ██████  ███████ ██   ██
#
# Initializing the dataset from folders
# ------------------------------------------------
@tf.function(experimental_follow_type_hints=True, jit_compile=True)
def datasetLoad( self ):
        rootfolder = self.data_dir
        # FOR SHMGAN Dataset
        # path1 = os.path.join( rootfolder, 'I0' )
        # path2 = os.path.join( rootfolder, 'I45' )
        # path3 = os.path.join( rootfolder, 'I90' )
        # path4 = os.path.join( rootfolder, 'I135' )
        # path5 = os.path.join( rootfolder, 'ED' )

        # FOR PSD Polar Dataset
        path1 = os.path.join( rootfolder, 'I0' )
        path2 = os.path.join( rootfolder, 'I60' )
        path3 = os.path.join( rootfolder, 'I90' )
        path4 = os.path.join( rootfolder, 'I150' )
        path5 = os.path.join( rootfolder, 'ED' )

        data_dir1 = pathlib.Path( path1 )
        data_dir2 = pathlib.Path( path2 )
        data_dir3 = pathlib.Path( path3 )
        data_dir4 = pathlib.Path( path4 )
        data_dir5 = pathlib.Path( path5 )

        # Intialize array for saving each image's values
        self.stddev_arr   = []
        self.mean_arr     = []
        self.variance_arr = []

        # NOTE:  => The generated datasets do not have any lables

        train_ds_0 = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir1 ),
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
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch(25)
                # .map(lambda x: self.custom_per_image_standardization(x), num_parallel_calls=tf.data.AUTOTUNE  ) \
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        # Manually update the labels (?)
        train_ds_0.class_names = 'I0'

        train_ds_45 = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir2 ),
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
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch(25)
                # .map(lambda x: self.custom_per_image_standardization(x), num_parallel_calls=tf.data.AUTOTUNE  ) \
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        train_ds_45.class_names = 'I45'

        train_ds_90 = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir3 ),
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
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch( 25)
                # .map(lambda x: self.custom_per_image_standardization(x), num_parallel_calls=tf.data.AUTOTUNE  ) \
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        train_ds_90.class_names = 'I90'

        train_ds_135 = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir4 ),
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
                .map(lambda x: (x / 255.0), num_parallel_calls=tf.data.AUTOTUNE  ) \
                .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch(25)
                # .map(lambda x: self.custom_per_image_standardization(x), num_parallel_calls=tf.data.AUTOTUNE  ) \
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE  ) \
        train_ds_135.class_names = 'I135'

        train_ds_ED = tf.keras.preprocessing.image_dataset_from_directory(
                str( data_dir5 ),
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
                .map(lambda x: x if self.random_flip else tf.image.flip_up_down( x ), num_parallel_calls=tf.data.AUTOTUNE) \
                .prefetch(25)
                # .map(lambda x: self.custom_per_image_standardization(x), num_parallel_calls=tf.data.AUTOTUNE  ) \
                # .map(lambda x: tf.image.per_image_standardization( x ) ) \
                # .map(lambda x: ((x / 127.5) - 1 ), num_parallel_calls=tf.data.AUTOTUNE ) \
        train_ds_ED.class_names = 'ED'


        # ZIP the datasets into one dataset
        loadedDataset = tf.data.Dataset.zip ( ( train_ds_0, train_ds_45, train_ds_90, train_ds_135, train_ds_ED ) )
        # dataset.map(time_consuming_mapping).cache().map(memory_consuming_mapping)

        # Sauce: https://www.tensorflow.org/guide/data_performance_analysis#3_are_you_reaching_high_cpu_utilization
        options = tf.data.Options()
        options.threading.max_intra_op_parallelism = 1
        loadedDataset = loadedDataset.with_options(options)


        # -------------------------------------------------------
        # Repeat/parse the loaded dataset for the same number as epochs...
        # cahces and prefetches the datasets for performance
        # repeat for epochs
        self.loadedDataset = loadedDataset.cache().repeat( self.num_epochs ).prefetch( buffer_size =25)
        # -------------------------------------------------------

        # return the number of files loaded , to calculate iterations per batch
        self.length_dataset = len(np.concatenate([i for i in train_ds_0], axis=0))
        # returns the zipped dataset for use with iterator
        return self.length_dataset, self.loadedDataset
