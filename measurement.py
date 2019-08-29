import numpy
import math
import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def SSIM(img1, img2):
    return ssim(img1, img2, multichannel=True)
