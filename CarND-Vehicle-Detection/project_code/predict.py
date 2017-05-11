from hog_subsampling import find_cars
import glob
import pickle
import matplotlib.image as mpimg
from sklearn.externals import joblib
import cv2
import numpy as np

def get_bboxes(img):
  [colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range] = pickle.load(open("variables_dump.p", "rb"))
  svc = pickle.load(open('svc_dump.pkl', 'rb'))
  X_scaler = pickle.load(open('X_scaler_dump.pkl', 'rb'))
  
  find_car_iterations = [[400, 460, 1.0], [400, None, 1.5], [400, None, 2.0], [400, None, 2.5]]
  # find_car_iterations = [[400, 700, 2.5]]
  find_car_bboxes = np.array([])
  for find_car_iteration in find_car_iterations:
    [ystart, ystop, scale] = find_car_iteration
    bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hist_range, colorspace)
    bboxes = np.array(bboxes)
    if find_car_bboxes.size == 0:
      find_car_bboxes = bboxes
    elif bboxes.size != 0:
      find_car_bboxes = np.concatenate([find_car_bboxes, bboxes], axis=0)

  return find_car_bboxes


def display_bboxes():
  image_names = glob.glob('../test_images/test*.jpg')
  for index, image_name in enumerate(image_names):
    img = mpimg.imread(image_name)
    find_car_bboxes = get_bboxes(img)
    out_img = np.copy(img)
    for find_car_bbox in find_car_bboxes:
      out_img = cv2.rectangle(out_img, (find_car_bbox[0][0], find_car_bbox[0][1]), (find_car_bbox[1][0], find_car_bbox[1][1]), (0,0,255), 6)
    print('Saving predicted image' + str(index))
    mpimg.imsave('../test_output/test' + str(index) + '.png', out_img)  

# display_bboxes()
