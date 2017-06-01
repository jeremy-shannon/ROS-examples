#!/usr/bin/env python
import rospy
from generate_tracklet import *
from lidar.msg import img_with_pose
import json
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
import h5py

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
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model.h5')
graph = tf.get_default_graph()

poses = []

def writeTracklet():
    collection = TrackletCollection()
    obs_tracklet = Tracklet(
        object_type='Car', l=4.2, w=1.5, h=1.5, first_frame=0)
    obs_tracklet.poses = poses
    collection.tracklets.append(obs_tracklet)
    #tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
    collection.write_xml('tracklet_labels.xml')   

def carPredicted(data):
    global graph
    with graph.as_default():
        img = np.resize(data, (1,64,64,1))
        prediction = model.predict(img, 1)
        if (prediction[0][1] > 0.5):
            return True
        else:
            return False                    

def callback(data):
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data.img, "mono8")
    except CvBridgeError as e:
        print(e)
    #if (carPredicted(cv_image)):
    poses.append(dict(tx=data.pose.x, ty=data.pose.y, tz=-0.75, rx=0, ry=0, rz=0))
    
def listener():
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    rospy.init_node('listener', anonymous=True)
    print("classifier and tracklet generator node started")
    rospy.Subscriber("/heightmap/cluster_and_pose", img_with_pose, callback)
    rospy.spin()
    rospy.on_shutdown(writeTracklet)

if __name__ == '__main__':
    listener()