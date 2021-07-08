# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
from PIL import Image
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.misc

from IPython.display import display
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Lambda, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import csv

import utils # from utils.py

#%matplotlib inline

kSEED = 5
SIDE_STEERING_CONSTANT = 0.25
NUM_BINS = 23
NB_EPOCH = 30
LR = 1e-3

Center_Image = []
Left_Image = []
Right_Image = []
Steering_Angle = []
Throttle = []
Break = []
Speed = []

file_name = 'driving_log_2021.csv'

with open(file_name) as f:

    reader = csv.reader(f)

    for row in reader:
        Center_Image.append(str(row[0].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
        Left_Image.append(str(row[1].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
        Right_Image.append(str(row[2].replace('/home/deepstation/Simulator/App/IMG_2021/', '')))
        Steering_Angle.append(str(row[3]))
        Throttle.append(str(row[4]))
        Break.append(str(row[5]))
        Speed.append(str(row[6]))

angle_before = 0
action = []
l = len(Steering_Angle)
#print(l)
for i in range(l):
    angle = Steering_Angle[i] - angle_before
    angle_before = Steering_Angle[i]
    #print(Steering_Angle[85],Throttle[85])
    if angle < 0 and Throttle[i] >= 0:
        action.append(0)

    elif angle > 0 and Throttle[i] >= 0:
        action.append(1)

    elif angle == 0 and Throttle[i] > 0:
        action.append(2)

    elif angle == 0 and Throttle[i] == 0:
        action.append(3)

data_1 = []
data = []
for i in range(l):
    #data = np.array(l)
    data = np.append(data,Center_Image[i])
    #Input.append(Center_Image[i])
    #Input.append(Steering_Angle[i])
    #Input.append(Throttle[i])
    #data = np.append(data,Steering_Angle[i])
    #data = np.append(data,Throttle[i])
    #data_1.append(data)

X_train, X_val, y_train, y_val = train_test_split(data, action, test_size=0.15, random_state=kSEED)

def batch_generator(data, labels, augment_data=True, batch_size=32):
    """
    Keras Batch Generator to create a generator of training examples for model.
    :param images: Training image data.
    :param angles: Angle data for images.
    :param batch_size: Batch size of each training run.
    :param augment_data: If the data should be augmented.
    :return: A batch generator.
    """

    batch_action = []
    batch_data = []
    batch_images = []
    sample_count = 0
    batch_data_1 = []
    batch_action_1 = []

    while True:
        # Shuffle indices to minimize overfitting.
        for i in np.random.permutation(len(data)):

            # Image (1) -> Center image and steering angle.
            center_path = data[i]
            #print(center_path)
            #angle = float(data[i][1])
            #throttle = float(data[i][2])
            action = labels[i]

            center_image = utils.load_image(center_path)
            #batch_images = np.expand_dims(center_image,axis=0)
            batch_data.append(center_image)
            #batch_data.append(angle)
            #batch_data.append(throttle)
            #batch_data_1.append(batch_data)
            batch_action.append(action)
            #batch_action_1.append(batch_action)

            #print(batch_data_1[0][1])

            sample_count += 1

            if(augment_data):
                if sample_count % 2 == 0:
                    center_image, angle = utils.jitter_image(center_path,angle)
                    #batch_images = np.expand_dims(center_image,axis=0)
                    batch_data.append(center_image)
                    #batch_data.append(angle)
                    #batch_data.append(throttle)
                    #batch_data_1.append(batch_data)
                    batch_action.append(action)
                else:
                    center_image= utils.tint_image(center_path)
                    #batch_images = np.expand_dims(center_image,axis=0)
                    batch_data.append(center_image)
                    #batch_data.append(angle)
                    #batch_data.append(throttle)
                    #batch_data_1.append(batch_data)
                    batch_action.append(action)

                sample_count += 1

            if ((sample_count % batch_size == 0) or (sample_count % len(data) == 0)):
                yield np.array(batch_data), np.array(batch_action)
                # Reset batch/
                batch_data = []
                batch_action = []


model = Sequential()
# Lambda layer normalizes pixel values between 0 and 1
model.add(Dense(32, input_shape=(80, 320, 3)))
# Convolutional layer (1)
model.add(Conv2D(24, (5,5), padding='same', activation='relu', strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
# Convolutional layer (2)
model.add(Conv2D(36, (5,5), padding='same', activation='relu', strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
# Convolutional layer (3)
model.add(Conv2D(48, (5,5), padding='same', activation='relu', strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
# Convolutional layer (4)
model.add(Conv2D(64, (3,3), padding='same', activation='relu', strides=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
# Convolutional layer (5)
model.add(Conv2D(64, (3,3), padding='same', activation='relu', strides=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
# Flatten Layer
model.add(Flatten())
# Dense Layer (1)
model.add(Dense(1164, activation='relu'))
# Dense layer (2)
model.add(Dense(100, activation='relu'))
# Dense layer (3)
model.add(Dense(50, activation='relu'))
# Dense layer (4)
model.add(Dense(10, activation='relu'))
# Dense layer (5)
model.add(Dense(4, activation='softmax'))
# Compile model
model.compile(optimizer=Adam(lr=LR, decay=LR / NB_EPOCH), loss='mse')

generator_train = batch_generator(X_train, y_train, augment_data=True)
generator_val = batch_generator(X_val, y_val, augment_data=False)

model.fit_generator(
        generator_train,
        steps_per_epoch=3 * len(X_train),
        epochs=10,
        validation_data=generator_val,
        validation_steps=len(X_val),
        verbose=1)
