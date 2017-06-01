from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
import math
# Fix error with TF and Keras
import tensorflow as tf
# tf.python.control_flow_ops = tf

# load training data
car_image_files = glob.glob('images/car/*.png')
noncar_image_files = glob.glob('images/noncar/*.png')

def generate_training_data(batch_size=1500):
    '''
    method for the model training data generator to load images, then yield them to the model. 
    '''
    X,y = ([],[])
    while True:       
        # pick a random car image
        img_file = car_image_files[np.random.choice(len(car_image_files))]
        img = cv2.imread(img_file, 0)
        img = np.resize(img, (64,64,1))
        X.append(img)
        y.append([0,1])
        # pick a random noncar image
        img_file = noncar_image_files[np.random.choice(len(noncar_image_files))]
        img = cv2.imread(img_file, 0)
        img = np.resize(img, (64,64,1))
        X.append(img)
        y.append([1,0])
        if len(X) == batch_size:
            X, y = shuffle(X, y)
            yield (np.array(X), np.array(y))
            X, y = ([],[])


model = Sequential()

# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,1)))

# Add three 5x5 convolution layers (output depth 6, 16, and 24), each with 2x2 stride
model.add(Convolution2D(6, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(16, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
model.add(ELU())

#model.add(Dropout(0.50))

# Add a flatten layer
model.add(Flatten())

# Add two fully connected layers (depth 100, 50), elu activation (and dropouts)
model.add(Dense(100, kernel_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(50, kernel_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(2, kernel_initializer='uniform'))
model.add(Activation('softmax'))

# Compile and train the model, 
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# initialize generator
train_gen = generate_training_data()
validation_gen = generate_training_data()

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

history = model.fit_generator(train_gen, validation_data=validation_gen, 
    steps_per_epoch=10, validation_steps=2, epochs=10, verbose=2, callbacks=[checkpoint])
# print('Test Loss:', model.evaluate_generator(test_gen, 128))

print(model.summary())

# y_pred = model.predict(X_test, n, verbose=2)

# Save model data
model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)