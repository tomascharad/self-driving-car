import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from hog_classify import train
import random

cars = glob.glob('../vehicles/*/image*.png')
notcars = glob.glob('../non-vehicles/*/image*.png')

colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 16
pix_per_cell = 8
cell_per_block = 4
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 16
hist_range=(0, 256)
print('Training')

train(cars, notcars, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

print('Finish training')