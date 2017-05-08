from hog_subsampling import find_cars
import glob
import pickle
import matplotlib.image as mpimg
from sklearn.externals import joblib
import cv2

image_names = glob.glob('../test_images/test*.jpg')
for index, image_name in enumerate(image_names):
  img = mpimg.imread(image_name)
  [colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range] = pickle.load(open("variables_dump.p", "rb"))
  print([colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range])
  svc = joblib.load('svc_dump.pkl')
  X_scaler = joblib.load('X_scaler_dump.pkl')
  ystart = 300
  ystop = None
  scale = 1.5
  [out_img, bboxes] = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hist_range, colorspace, visualize=True)
  print('Saving predicted image')
  mpimg.imsave('../test_output/test' + str(index) + '.png', out_img)