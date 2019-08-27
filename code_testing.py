# Code testing: where to learn python and test code
import os, os.path
from random import sample
import matplotlib.pyplot as plt
import cv2
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

# Check tensor flow GPU
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

img = cv2.imread('./data_lr/lena_low.png')

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# plt.subplot(1, 1, 1)
# plt.imshow(gray, interpolation='nearest')
# plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='bicubic')
plt.axis('off')

plt.show()