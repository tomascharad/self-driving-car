import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


hog_classify()




image_names = glob.glob('../test_images/test*.jpg')
images = []
for image_name in image_names:
    img = cv2.imread(image_name)
    images.append(img)
    plt.imshow(img)



