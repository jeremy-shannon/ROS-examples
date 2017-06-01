from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffleimport matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
import math
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


car_image_files = glob.glob('images/car/*.png')
noncar_image_files = glob.glob('images/noncar/*.png')
car_images = []
noncar_images = []
for file in car_image_files:
    car_images.append(mpimg.imread(file))
for file in noncar_image_files:
    noncar_images.append(mpimg.imread(file))

print(len(car_images), len(noncar_images))
print(car_images[1].shape)




model = Sequential()

# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,1)))

# Add three 5x5 convolution layers (output depth 6, 16, and 24), each with 2x2 stride
model.add(Convolution2D(6, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

#model.add(Dropout(0.50))

# Add a flatten layer
model.add(Flatten())

# Add two fully connected layers (depth 100, 50), elu activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
#model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(2))

# Compile and train the model, 
model.compile(optimizer=Adam(lr=1e-4), loss='mse')

# initialize generators
train_gen = generate_training_data(image_paths_train, angles_train, validation_flag=False, batch_size=64)
val_gen = generate_training_data(image_paths_train, angles_train, validation_flag=True, batch_size=64)
test_gen = generate_training_data(image_paths_test, angles_test, validation_flag=True, batch_size=64)

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

#history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2, verbose=2)
history = model.fit_generator(train_gen, validation_data=val_gen, nb_val_samples=2560, samples_per_epoch=23040, 
                                nb_epoch=5, verbose=2, callbacks=[checkpoint])
# print('Test Loss:', model.evaluate_generator(test_gen, 128))

print(model.summary())

# # visualize some predictions
# n = 12
# X_test,y_test = generate_training_data_for_visualization(image_paths_test[:n], angles_test[:n], batch_size=n,                                                                     validation_flag=True)
# y_pred = model.predict(X_test, n, verbose=2)
# visualize_dataset(X_test, y_test, y_pred)

# Save model data
model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)