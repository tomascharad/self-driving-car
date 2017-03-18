import numpy as np
import csv

print('Starting')

samples = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
# TCT: Removing headers
samples = samples[1:]

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import sklearn

correction = 0.2

def augment_brighnes(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert it to hsv
  hsv[2] = hsv[2] + np.random.randint(-10, 10, dtype=np.int8) # change brightness of images (by -10 to 10 of original value)
  hsv[2] = hsv[2] * np.random.randint(5, 15, dtype=np.uint8) / 10  # change brightness of images (50-150% of original value)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    sklearn.utils.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []
      for batch_sample in batch_samples:
        center_angle = float(batch_sample[3])
        for i in range(3):
          actual_steering = center_angle
          if i == 1:
            actual_steering = actual_steering + correction
          if i == 2:
            actual_steering = actual_steering - correction
          name = './data/IMG/'+batch_sample[i].split('/')[-1]
          # print(name)
          image = cv2.imread(name)

          images.append(image)
          angles.append(actual_steering)
          # TCT: Augmentation
          images.append(cv2.flip(image,1))
          angles.append(actual_steering * -1.0)

          images.append(augment_brighnes(image))
          angles.append(actual_steering)
          # TCT: brightness
          


      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('Defining model')
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(25, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

print('Fitting model')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 3, nb_epoch=3)
model.save('model_brightness.h5')
print('Finishing saving model')

import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# print('The example shows')
# print('loss: 3.6950 - val_loss: 0.7409')