import rospy
import numpy as np
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int32


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
        self.red_fruit_pub = rospy.Publisher("/red_fruit_arr_", MarkerArray, queue_size=10)
        self.red_fruit_count_pub = rospy.Publisher('/red_fruit_count', Int32, queue_size=10, latch=True)
        self.red_fruit_arr_ = MarkerArray()
        self.yellow_fruit_pub = rospy.Publisher("/yellow_fruit_arr_", MarkerArray, queue_size=10)
        self.yellow_fruit_count_pub = rospy.Publisher('/yellow_fruit_count', Int32, queue_size=10, latch=True)
        self.yellow_fruit_arr_ = MarkerArray()

        self.green_dist = 1.0
        self.red_dist = 1.18
        self.yellow_dist = 1.5

    def add_red_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll, two_d_size):
        print("add_fruit_marker() called")
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        fruit_size = two_d_size/10
        marker.scale.x = fruit_size
        marker.scale.y = fruit_size
        marker.scale.z = fruit_size
        marker.color.a = 0.5
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("fruit pose isnan, return")
            return

        position[0] = round(position[0],0)

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.id = fruit_id

        rpy_roll = abs(rpy_roll)

        for i in range(0,len(self.red_fruit_arr_.markers)):
            old_marker = self.red_fruit_arr_.markers[i]
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= self.red_dist):
                print("duplicate red_fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                old_marker.color.a = old_marker.color.a + 0.5
                old_marker.scale.x = (3*old_marker.scale.x + fruit_size)/4
                old_marker.scale.y = (3*old_marker.scale.y + fruit_size)/4
                old_marker.scale.z = (3*old_marker.scale.z + fruit_size)/4
                # self.red_fruit_count_pub.publish(i+1)
                return i+1
        # if it's a new red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.red_fruit_arr_.markers.append(marker)
        new_red_id = len(self.red_fruit_arr_.markers)
        self.red_fruit_count_pub.publish(new_red_id)
        return new_red_id

    def add_yellow_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll, two_d_size):
        print("add_fruit_marker() called")
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        fruit_size = two_d_size/10
        marker.scale.x = fruit_size
        marker.scale.y = fruit_size
        marker.scale.z = fruit_size
        marker.color.a = 0.5
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("fruit pose isnan, return")
            return

        position[0] = round(position[0],0)

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.id = fruit_id

        rpy_roll = abs(rpy_roll)

        for i in range(0,len(self.yellow_fruit_arr_.markers)):
            old_marker = self.yellow_fruit_arr_.markers[i]
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= self.yellow_dist):
                print("duplicate yellow_fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                old_marker.color.a = old_marker.color.a + 0.5
                old_marker.scale.x = (3*old_marker.scale.x + fruit_size)/4
                old_marker.scale.y = (3*old_marker.scale.y + fruit_size)/4
                old_marker.scale.z = (3*old_marker.scale.z + fruit_size)/4
                # self.yellow_fruit_count_pub.publish(i+1)
                print(f"yellow_id by IIR is: {i+1}, start from 1, NOT 0")
                return i+1
        # if it's a new yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.yellow_fruit_arr_.markers.append(marker)
        new_yellow_id = len(self.yellow_fruit_arr_.markers)
        self.yellow_fruit_count_pub.publish(new_yellow_id)
        print(f"new_yellow_id is: {new_yellow_id}")
        return new_yellow_id




    def publish_markers(self):
        # print("publish from class===================================================")
        self.red_fruit_pub.publish(self.red_fruit_arr_)
        self.yellow_fruit_pub.publish(self.yellow_fruit_arr_)


