import io
import os
import cv2
import numpy as np 
import pandas as pd 
import math 
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from sklearn.metrics import confusion_matrix
from PIL import Image

#open csv file containind CK+ data

with open("updated_data.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("no. of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[0].split(" ")))

#separate train and test sets

x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1, num_of_instances):
    try:
        img, emotion, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, "float32")
        emotion = keras.utils.to_categorical(emotion, 5) #one-hot encoding

        if "Training" in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif "Test" in usage:
            y_test.append(emotion)
            x_test.append(pixels)

        # print(i)
    except:
        print("", end="")

#transform --> numpy array

x_train = np.array(x_train, "float32")
y_train = np.array(y_train, "float32")
x_test = np.array(x_test, "float32")
y_test = np.array(y_test, "float32")

print("x_train: ", x_train.shape[0])
print("x_test: ", x_test.shape[0])

print("y_train: ", y_train.shape[0])
print("y_test: ", y_test.shape[0])

x_train /= 255 #normalise
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 275, 496, 1)
x_train = x_train.astype("float32")
x_test = x_test.reshape(x_test.shape[0], 275, 496, 1)
x_test = x_test.astype("float32")

print(x_train.shape[0], 'train samples') #...gives total samples = 467 i.e. two are missing?
print(x_test.shape[0], 'test samples')

#attempt to make CNN

model = Sequential()

#conv layer 1
model.add(Conv2D(64, (7, 7), activation='relu', input_shape=[275,496,1])) 
model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

#conv layer 2
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#conv layer 3
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

#conv layer 4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(Flatten())

#NN
model.add(Dense(512, activation='relu', kernel_regularizer='l2')) #original 1st param = 1024
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_regularizer='l2')) #original 1st param = 1024
model.add(Dropout(0.2))

model.add(Dense(5, activation='softmax')) #5 classes

#batch
batch_size = 43 
steps_per_epoch = len(x_train)//batch_size

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size  = batch_size) 

opt = keras.optimizers.Adam()

model.compile(loss="categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

epochs = 6
fit = True

if fit == True:
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)


#evaluate!
score  = model.evaluate(x_train, y_train, verbose=0)
print(model.metrics_names)
print("Train loss: ", score[0])
print("Train accuracy: ", 100*score[1])

score  = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print("Test loss: ", score[0])
print("Test accuracy: ", 100*score[1])


img = image.load_img("cropped_img.png", grayscale=True, target_size=(275, 496))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

pred = model.predict(x)
print(pred)

maximum = np.max(pred[0])
max_index = np.where(pred[0] == maximum)

print("index: ", max_index)

if (max_index[0] == 0):
    emotion = "NEUTRAL"
elif (max_index[0] == 1):
    emotion = "ANGRY"
elif (max_index[0] == 2):
    emotion = "HAPPY"
elif (max_index[0] == 3):
    emotion = "SAD"
elif (max_index[0] == 4):
    emotion = "SURPRISED"

print(emotion)