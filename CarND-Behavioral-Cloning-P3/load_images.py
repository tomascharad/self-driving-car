import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

images = []
measurements = []
print('Loading images')
for line in lines[1:]:
  correction = 0.2
  steering_center = float(line[3])
  for i in range(3):
    actual_steering = steering_center
    if i == 1:
      actual_steering = actual_steering + correction
    if i == 2:
      actual_steering = actual_steering - correction
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(actual_steering)
print('Finish loading images')

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)