#!/usr/bin/env python
# title           :Utils.py
# description     :Have helper functions to process images and plot images
# author          :Deepak Birla
# date            :2018/10/30
# usage           :imported in other files
# python_version  :3.5.4

from keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
from scipy.misc import imresize
import os
import sys
import cv2

import measurement as ms

import matplotlib.pyplot as plt

plt.switch_backend('agg')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = array(images)
    return images_hr


# Takes list of images and provide LR images in form of numpy array
# Using imresize to down sampling images
def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        images.append(
            imresize(images_real[img], [images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale],
                     interp='bicubic', mode=None))
    images_lr = array(images)
    return images_lr


#  normalize images to range [-1, 1]
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext, image_shape):
    files = []
    file_names = []
    count = 0
    (width, height, n) = image_shape
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = data.imread(os.path.join(d, f))
                if len(image.shape) > 2:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    files.append(image)
                    file_names.append(os.path.join(d, f))
                else:
                    file_name = os.path.join(d, f)
                    print('Image', file_name, 'is not 3 dimension')
                count = count + 1
    return files

def load_data_from_dirs_for_LRTest(dirs, ext, image_shape):
    files = []
    file_names = []
    count = 0
    (width, height, n) = image_shape
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = data.imread(os.path.join(d, f))
                if len(image.shape) > 2:
                    image = cv2.resize(image, (int(width/4), int(height/4)), interpolation=cv2.INTER_AREA)
                    files.append(image)
                    file_names.append(os.path.join(d, f))
                else:
                    file_name = os.path.join(d, f)
                    print('Image', file_name, 'is not 3 dimension')
                count = count + 1
    return files


def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files


def load_training_data(directory, ext, image_shape, number_of_images=1000, train_test_ratio=0.8):
    print("========= Start loading data ==========")
    number_of_train_images = int(number_of_images * train_test_ratio)

    print(number_of_train_images)

    files = load_data_from_dirs(load_path(directory), ext, image_shape)

    print(len(files))

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    test_array = array(files)
    if len(test_array.shape) < 3:
        print("Images are of not same shape")
        print("Please provide same shape images")
        sys.exit()

    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]

    # Re-scale image to x4 of image shape was defined

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)

    x_train_lr = lr_images(x_train, 4)
    x_train_lr = normalize(x_train_lr)

    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(x_test, 4)
    x_test_lr = normalize(x_test_lr)

    print("========= End loading data ==========")

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


# Load HR images
def load_test_data_for_model(directory, ext, image_shape, number_of_images=100):
    files = load_data_from_dirs(load_path(directory), ext, image_shape)

    print("Load HR image from ", directory, "successfully")

    for file in files:
        print(file.shape)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, x_test_hr


# Load LR images
def load_test_data(directory, ext, image_shape, number_of_images=100):
    files = load_data_from_dirs_for_LRTest(load_path(directory), ext, image_shape)

    print("Load LR image from ", directory, "successfully")

    for file in files:
        print(file.shape)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    # x_test_lr = lr_images(files, 4)
    # Show dimention of LR images
    x_test_lr = normalize(array(files))

    return x_test_lr


# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)

    # plt.show()


# Plots and save generated images(in form LR, SR, HR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr, dim=(1, 5), figsize=(50, 8)):
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    # Label measurement
    label = 'PSNR: %2.f, SSIM: %.2f'

    for index in range(examples):
        plt.figure(figsize=figsize)

        plt.subplot(dim[0], dim[1], 1)
        nearest_img = cv2.resize(image_batch_lr[index], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        m = ms.PSNR(nearest_img, image_batch_hr[index])
        s = ms.SSIM(nearest_img, image_batch_hr[index])
        plt.gca().set_title('Nearest neighbor '+ label % (m, s), fontsize=25)
        # cv2.imwrite('SRGAN_output/NN_image_only_%d.png' % index,
        #             cv2.cvtColor(nearest_img, cv2.COLOR_BGR2RGB))
        plt.imshow(nearest_img, interpolation='none')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        bilinear_img = cv2.resize(image_batch_lr[index], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        m = ms.PSNR(bilinear_img, image_batch_hr[index])
        s = ms.SSIM(bilinear_img, image_batch_hr[index])
        plt.gca().set_title('Bilinear ' + label % (m, s), fontsize=25)
        # cv2.imwrite('SRGAN_output/BL_image_only_%d.png' % index,
        #             cv2.cvtColor(bilinear_img, cv2.COLOR_BGR2RGB))
        plt.imshow(bilinear_img, interpolation='none')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        bicubic_img = cv2.resize(image_batch_lr[index], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        m = ms.PSNR(bicubic_img, image_batch_hr[index])
        s = ms.SSIM(bicubic_img, image_batch_hr[index])
        plt.gca().set_title('Bicubic ' + label % (m, s), fontsize=25)
        # cv2.imwrite('SRGAN_output/Bicubic_image_only_%d.png' % index,
        #             cv2.cvtColor(bicubic_img, cv2.COLOR_BGR2RGB))
        plt.imshow(bicubic_img, interpolation='none')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 4)
        m = ms.PSNR(generated_image[index], image_batch_hr[index])
        s = ms.SSIM(generated_image[index], image_batch_hr[index])
        plt.gca().set_title('SRGANs ' + label % (m, s), fontsize=25)
        plt.imshow(generated_image[index], interpolation='none')
        cv2.imwrite('SRGAN_output/generated_image_only_%d.png' % index, cv2.cvtColor(generated_image[index], cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 5)
        plt.gca().set_title('Original', fontsize=25)
        plt.imshow(image_batch_hr[index], interpolation='none')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)

        # plt.show()


# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):
    examples = x_test_lr.shape[0]
    image_batch_lr = denormalize(x_test_lr)
    gen_img = generator.predict(x_test_lr)
    generated_image = denormalize(gen_img)

    for index in range(examples):
        # plt.figure(figsize=figsize)

        nearest_img = cv2.resize(image_batch_lr[index], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('output/NN_image_only_%d.png' % index,
                    cv2.cvtColor(nearest_img, cv2.COLOR_BGR2RGB))

        cv2.imwrite('output/high_res_result_image_%d.png' % index,
                    cv2.cvtColor(generated_image[index], cv2.COLOR_BGR2RGB))

        # plt.imshow(generated_image[index], interpolation='nearest')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)

        # plt.show()
