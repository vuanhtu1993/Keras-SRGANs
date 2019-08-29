# Code testing: where to learn python and test code
import os, os.path
from random import sample

# [File counting]
# for root, dirs, files in os.walk("./data", topdown=False):
#     print(len(files))

# [Delete files]
# files = os.listdir('./data')
# # output: List
# for file in sample(files, 2):
#     os.remove('./data/' + file)

# # [Get size(byte) image]
# files = os.listdir('./data/')
# # Sort image by size
# files_sorted = sorted(files, key=lambda file_: os.path.getsize('./data_hr/' + file_), reverse=True)

#

# # Move 3 largest size to other folder
# for file in files_sorted:
#     print(os.path.getsize('./data_hr/' + file))

# dir = os.fsencode('VN-celeb')
# for d in dir:
#     os.chdir(d)
#
#     dir2 = os.fsencode()

# list = os.listdir('./VN_dataset/')
#
# print(len(list))

# import tensorflow as tf
# from tensorflow.python.client import device_lib
#
#
# print(device_lib.list_local_devices())
# print("GPU Available: ", tf.test.is_gpu_available())
#
# print("Device name:", tf.test.gpu_device_name())
#
# print(tf.__version__)

import measurement
import matplotlib.pyplot as plt
import cv2
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = measurement.psnr_measure(imageA, imageB)
    s = measurement.ssim_measure(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()

original = cv2.imread("data_hr/66.png")
contrast = cv2.imread("data_hr/66.png")

bilinear_img = cv2.resize(original, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)


compare_images(bilinear_img, bilinear_img, "test")