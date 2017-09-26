#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:08:36 2017

@author: vishal
"""


import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
correction = 0.3 # this is a parameter to tune
from keras import optimizers

import matplotlib.pyplot as plt


#Load CSV file
lines = []
with open('/home/vishal/SDCND/CarND-Behavioral-Cloning-P3/data/Customdata/track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#generator for preprocessing images 
def generator(samples, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path=batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/home/vishal/SDCND/CarND-Behavioral-Cloning-P3/data/Customdata/track1/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    if(i==0):
                        steering_center = float(batch_sample[3])
                        measurements.append(steering_center)
                    elif(i==1):
                        measurements.append(steering_center + correction)
                    else:
                        measurements.append(steering_center - correction)
            # trim image to only see section with road
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                 augmented_images.append(image)
                 augmented_measurements.append(measurement)
                 augmented_images.append(cv2.flip(image, 1))
                 augmented_measurements.append(measurement*-1.0)
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)                      
        
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda , Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5,input_shape=(160,320,3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation="relu"))
model.add(Dropout(0.2))

model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


exit()
