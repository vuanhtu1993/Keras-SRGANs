import cv2
import os.path
import sys
import matplotlib.pyplot as plt


def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def resize_image(img_src, dimension=(96, 96)):
    print(load_path(img_src))
    # img = cv2.imread(load_path(img_src), cv2.IMREAD_UNCHANGED)
    # resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Resized image", resized)


# Exampleq
# resize_image('../data_hr/')
im = cv2.imread("/Tools/the_girl.jpg")
print(im)
print(os.path)
