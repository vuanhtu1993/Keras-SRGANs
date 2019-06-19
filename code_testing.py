# Code testing: where to learn python and test code
import os
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

import os

dir = os.fsencode('VN-celeb')
for d in dir:
    os.chdir(d)

    dir2 = os.fsencode()