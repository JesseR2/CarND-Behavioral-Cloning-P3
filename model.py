
# coding: utf-8

import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

#load csv file list
samples = []
with open('t1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

shuffle(samples)

#split up train and val sets - no test set, track testing will be used instead
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))


# augmentations
def shift(image, angle, mx_shift): #translate left and right up to 25 pixels -adjust steer angle too
    mu, sigma = 0, mx_shift *.3 # mean and standard deviation
    offset = int(np.random.normal(mu, sigma))
    if offset < -25:
        offset = -25
    if offset > 25:
        offset = 25
    
    rows, cols, col = image.shape
    M = np.float32([[1,0,offset],[0,1,0]])
    image = cv2.warpAffine(image,M,(cols, rows))
    
    angle = angle + (2 / 25 * (offset / 25))
    
    return image, angle

def bright(image, min_br, max_br): # increase or decrease all 3 color channels
    
    br = np.random.rand()
    delta = max_br - min_br
    
    image = image * (min_br + delta * br) #try cv2 rgb2hsv, just adjust v channel
    
    return image

def flip(image, angle): #mirror image and negate steer angle
    if np.random.rand()>.5:
        image = cv2.flip(image,1)
        angle = - angle
    return image, angle


#generator load and augment data and yield it to Keras
def generator(samples, batch_size=32):
   
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = (".\\data\\" + batch_sample[0]).replace("\\","/") #ajust win path for linux
                image = cv2.imread(name)
                angle = float(batch_sample[1])
                
                image, angle = shift(image, angle, 25) #translate left and right
                
                image = bright(image, 0.8, 1.2) #perturb brightness
                
                image, angle = flip(image, angle) #mirror images
                
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#define generators

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# define model - similar to Nvidia model but uses maxPool rather than 2x strides on the CNNs, uses all 3x3 kernels, also uses RELU rather than ELU
drop_prob = 0.2

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(25,25))))

model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_prob))

model.add(Conv2D(36, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_prob))

model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_prob))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_prob))

model.add(Flatten())

model.add(Dense(1164, activation='relu'))
model.add(Dropout(drop_prob))

model.add(Dense(100, activation='relu'))
model.add(Dropout(drop_prob))

model.add(Dense(50, activation='relu'))
model.add(Dropout(drop_prob))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#train the model
history = model.fit_generator(train_generator, steps_per_epoch = len(train_samples)//32,
                    validation_data=validation_generator, validation_steps=len(validation_samples)//32, epochs=8)
#save the model
model.save('model.h5')


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

