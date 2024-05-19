import rospy
import numpy as np
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion



def calc_marker_dist(marker1, marker2):
    dx = marker1.pose.position.x - marker2.pose.position.x
    dy = marker1.pose.position.y - marker2.pose.position.y
    dz = marker1.pose.position.z - marker2.pose.position.z
    dist = math.sqrt(dx*dx+ dy*dy + dz*dz)

    # print("calc_marker_dist() dx:", dx)
    # print("calc_marker_dist() dy:", dy)
    # print("calc_marker_dist() dz:", dz)
    # print("calc_marker_dist() dist is:", dist)

    if math.isnan(dist):
        print("dist is nan")
        return 0
    return dist



class PlantFruitDatabase:
    def __init__(self):
        self.fruit_pub = rospy.Publisher("fruit_arr_", MarkerArray, queue_size=10)
        self.fruit_arr_ = MarkerArray()

        self.green_dist = 1.0
        self.red_dist = 0.18
        self.yellow_dist = 1.1

    def add_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll, two_d_size):
        print("add_fruit_marker() called")
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        yellow_size = two_d_size/5
        marker.scale.x = yellow_size
        marker.scale.y = yellow_size
        marker.scale.z = yellow_size
        marker.color.a = 0.5
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("fruit pose isnan, return")
            return

        position[0] = round(position[0],0)
        # if(position[0]<4):
        #     position[0] = 3.5
        # elif(4.8 < position[0]):
        #     position[0] = 6.5
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.id = fruit_id

        rpy_roll = abs(rpy_roll)
        for old_marker in self.fruit_arr_.markers:
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= self.yellow_dist and fruit_color == "yellow"):
                print("duplicate fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                old_marker.color.a = old_marker.color.a + 0.5
                old_marker.scale.x = (3*old_marker.scale.x + yellow_size)/4
                old_marker.scale.y = (3*old_marker.scale.y + yellow_size)/4
                old_marker.scale.z = (3*old_marker.scale.z + yellow_size)/4
                return
            if (dist <= self.red_dist and fruit_color == "red"):
                print("duplicate fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                return

            

        if fruit_color == 'yellow':
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            self.fruit_arr_.markers.append(marker)
        elif fruit_color == 'red':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.fruit_arr_.markers.append(marker)
        elif fruit_color == 'purple':
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.5
            marker.type = marker.CYLINDER
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.3
            self.fruit_arr_.markers.append(marker)


    def publish_markers(self):
        # print("publish from class===================================================")
        self.fruit_pub.publish(self.fruit_arr_)


