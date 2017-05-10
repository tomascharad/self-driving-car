import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from hog_classify import train
import random

cars = glob.glob('../vehicles/*/image*.png')
# cars = glob.glob('../vehicles/GTI*/image*.png')
notcars = glob.glob('../non-vehicles/*/image*.png')

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
hist_range=(0, 2)
print('Training')

train(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

print('Finish training')