import matplotlib.image as mpimg
from lesson_functions import *
import glob

cars = glob.glob('../vehicles/*/image*.png')
notcars = glob.glob('../non-vehicles/*/image*.png')

car = mpimg.imread(cars[123])
notcar = mpimg.imread(notcars[123])

mpimg.imsave('../examples/car.png', car)
mpimg.imsave('../examples/notcar.png', notcar)

cspace = 'HSV'
orient = 16
pix_per_cell = 8
cell_per_block = 4

car_feature_image = convert_color(car, cspace=cspace)
not_feature_image = convert_color(notcar, cspace=cspace)

car_hog_features = []
for channel in range(car_feature_image.shape[2]):
    car_hog_features.append(get_hog_features(car_feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
car_hog_features = np.ravel(car_hog_features)

car_hog_features = []
for channel in range(car_feature_image.shape[2]):
    car_hog_features.append(get_hog_features(car_feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
car_hog_features = np.ravel(car_hog_features)

# TCT: Ended up here: TBD: set vis = True and visualize hog images

mpimg.imsave('../examples/car_hog_features.png', car_hog_features)
mpimg.imsave('../examples/notcar_hog_features.png', notcar_hog_features)


