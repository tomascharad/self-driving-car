import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from predict import get_bboxes
import glob

# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a 
# list of boxes for one of the images shown above
# box_list = pickle.load( open( "bbox_pickle.p", "rb" ))



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Add heat to each box in box list
# heat = add_heat(heat,box_list)
    
# # Apply threshold to help remove false positives
# heat = apply_threshold(heat,1)

# # Visualize the heatmap when displaying    
# heatmap = np.clip(heat, 0, 255)

# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(image), labels)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()
def sinlge_image_detection(img):
  find_car_bboxes = get_bboxes(img)
  heat = np.zeros_like(img[:,:,0]).astype(np.float)
  heatmap = add_heat(heat, find_car_bboxes)
  heatmap = apply_threshold(heatmap, 3)
  cliped_heatmap = np.clip(heat, 0, 255)
  labels = label(cliped_heatmap)
  draw_img = draw_labeled_bboxes(np.copy(img), labels)
  return [draw_img, cliped_heatmap]


def detect_multiple():
  image_names = glob.glob('../test_images/test*.jpg')
  for index, image_name in enumerate(image_names):
    img = mpimg.imread(image_name)
    [draw_img, cliped_heatmap] = sinlge_image_detection(img)
    print('Saving heatmap image' + str(index))
    mpimg.imsave('../test_output/test_labeled_heat' + str(index) + '.png', draw_img)
    mpimg.imsave('../test_output/test_heat' + str(index) + '.png', cliped_heatmap, cmap='hot')