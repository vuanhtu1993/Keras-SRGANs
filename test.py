#!/usr/bin/env python
# title           :test.py
# description     :to test the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python test.py --options
# python_version  :3.5.4

from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array

import os

from keras.models import load_model
from scipy.misc import imresize
import argparse

import Utils, Utils_model
from Utils_model import VGG_LOSS

# To fix error Initializing libiomp5.dylib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

image_shape = (384, 384, 3)


def test_model(input_hig_res, model, number_of_images, output_dir, image_shape, extension):
    print("-------- start est model for HR images processing -------")
    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, extension, image_shape, number_of_images)
    print('----- Finished pre-process image-----')
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)
    print('----- Finished generate image using model from ', output_dir, '-------')


def test_model_for_lr_images(input_low_res, model, number_of_images, output_dir, image_shape, extension):
    print("-------- Start test model for LR images processing ------------")
    x_test_lr = Utils.load_test_data(input_low_res, extension, image_shape, number_of_images)
    print('----- Finished pre-process image-----')
    Utils.plot_test_generated_images(output_dir, model, x_test_lr)
    print('----- Finished generate image using model from ', output_dir, '-------')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_lr_images',
                        help='Option to test model output or to test low resolution image')

    values = parser.parse_args()

    # Load loss define image shape
    loss = VGG_LOSS(image_shape)

    param_model_dir = './model/gen_model3000.h5'
    param_input_high_res = './data_hr/'
    param_input_low_res = './data_lr/'
    param_number_of_images = 1
    param_output_dir = './output/'
    param_extension = "jpg"

    # Load model (Keras)
    model = load_model(param_model_dir, custom_objects={'vgg_loss': loss.vgg_loss})

    if values.test_type == 'test_model':
        test_model(param_input_high_res,
                   model,
                   param_number_of_images,
                   param_output_dir,
                   image_shape,
                   param_extension)

    elif values.test_type == 'test_lr_images':
        test_model_for_lr_images(param_input_low_res,
                                 model,
                                 param_number_of_images,
                                 param_output_dir,
                                 image_shape,
                                 param_extension)

    else:
        print("No such option")
