#!/usr/bin/env python
import rospy
from generate_tracklet import *
from lidar.msg import img_with_pose

poses = []

def writeTracklet():
    collection = TrackletCollection()
    obs_tracklet = Tracklet(
        object_type='Car', l=4.2, w=1.5, h=1.5, first_frame=0)
    obs_tracklet.poses = poses
    collection.tracklets.append(obs_tracklet)
    #tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
    collection.write_xml('tracklet_labels.xml')    

def callback(data):
    poses.append(dict(tx=data.pose.x, ty=data.pose.y, tz=0.7, rx=0, ry=0, rz=0))
    
def listener():
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/heightmap/cluster_and_pose", img_with_pose, callback)
    rospy.spin()
    rospy.on_shutdown(writeTracklet)

if __name__ == '__main__':
    listener()