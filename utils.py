import itertools
import numpy as np
import cv2
import imutils
import random


def get_loader( filenames, labels, image_size = 128, batch_size = 16, mode = 'train' ):
    """Build and return a data loader."""
    n_batches = int( len( filenames ) / batch_size )
    total_samples = n_batches * batch_size

    for i in range( n_batches ):
        batch = filenames[i * batch_size: (i + 1) * batch_size]
        imgs = []
        for p in batch:
            image = cv2.imread( p )
            # image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
            # image = resize_images( image, image_size, image_size )
            if mode == 'train':
                proba = np.random.rand()
                if proba > 0.5:
                    image = cv2.flip( image, 1 )

            imgs.append( image )

        imgs = np.array( imgs ) / 127.5 - 1
        orig_labels = np.array( labels[i * batch_size:(i + 1) * batch_size] )
        target_labels = np.random.permutation( orig_labels )
        yield imgs, orig_labels, target_labels, batch


# ----------------------------------------
def preprocess( filenames_Itot, filenames_0deg, filenames_45deg, filenames_90deg, filenames_135deg, filenames_est_diffuse ):
    # We can fix the labels as follows: [itot, 0, 45, 90, 135, est_diff]
    # So we assign the labels to all the filenames and load each file per loop, to save memory
    # instead of reading all images and passing on the entire stack. Lots of images = much more mem req

    all_file_dataset = []
    all_file_labels = []
    test_dataset = []
    test_dataset_label = []
    train_dataset = []
    train_dataset_label = []
    a = []
    b = []

    # Stacking filenames and their labels as one-hot vectors
    for i in range( len( filenames_Itot ) ):
        all_file_dataset.append( filenames_Itot[i] )
        all_file_labels.append( [1, 0, 0, 0, 0, 0] )

    for i in range( len( filenames_0deg ) ):
        all_file_dataset.append( filenames_0deg[i] )
        all_file_labels.append( [0, 1, 0, 0, 0, 0] )

    for i in range( len( filenames_45deg ) ):
        all_file_dataset.append( filenames_45deg[i] )
        all_file_labels.append( [0, 0, 1, 0, 0, 0] )

    for i in range( len( filenames_90deg ) ):
        all_file_dataset.append( filenames_90deg[i] )
        all_file_labels.append( [0, 0, 0, 1, 0, 0] )

    for i in range( len( filenames_135deg ) ):
        all_file_dataset.append( filenames_135deg[i] )
        all_file_labels.append( [0, 0, 0, 0, 1, 0] )

    for i in range( len( filenames_est_diffuse ) ):
        all_file_dataset.append( filenames_est_diffuse[i] )
        all_file_labels.append( [0, 0, 0, 0, 0, 1] )

    # https://stackoverflow.com/questions/35201424/best-way-to-merge-a-2d-list-and-a-1d-list-in-python
    # Should return a 2-D matrix with first column as filenames and the lables appended as following cols
    # EXPECTED OUTPUT: # [ ['xyz.png'], [0, 0, 0, 0, 0, 1] ]
    # To access filename and label, we can use a 2D access variable ..
    # filelist [0]['index'] for filename
    # filelist [1]['index'] for label
    filelist = [x + [''] * (len( filenames_0deg ) - len( x )) for x in itertools.chain( [all_file_dataset], [all_file_labels] )]

    # shuffling the images along with their labels for training
    # Not working right now :/
    random.seed( 1234 )
    random.shuffle( filelist )
    #
    # if i < 3:
    #     test_dataset.append( all_file_dataset )
    #     test_dataset_label.append( all_file_dataset )
    # else:
    #     train_dataset.append( all_file_dataset )
    #     train_dataset_label.append( all_file_labels )

    test_dataset.append( all_file_dataset )
    test_dataset_label.append( all_file_labels )
    train_dataset.append( all_file_dataset )
    train_dataset_label.append( all_file_labels )

    # test_dataset_fix_label = create_labels(test_dataset_label, selected_attrs)
    # train_dataset_fix_label = create_labels(train_dataset_label, selected_attrs)

    # test_dataset_fix_label = create_labels( test_dataset_label, 1 )
    # train_dataset_fix_label = create_labels( train_dataset_label, 1 )

    print( '\n Finished preprocessing: Generated test and train datasets...' )
    # return test_dataset, test_dataset_label, train_dataset, train_dataset_label, test_dataset_fix_label, train_dataset_fix_label
    return test_dataset, test_dataset_label, train_dataset, train_dataset_label


# ----------------------------------------
def resize_images( img, rowsize, colsize ):
    # The loaded images will be resized to lower res for faster training and eval. After POC, higher res can be used
    # rows, cols, ch = img.shape
    # Adding white balance to remove the green tint generating from the polarized images
    resized_image = white_balance( imutils.resize( img, width = colsize, height = rowsize ) )
    return resized_image


# ----------------------------------------
def create_labels( c_org, selected_attrs = None ):
    pass


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