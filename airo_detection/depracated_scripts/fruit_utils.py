import rospy
import numpy as np
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion



def XYZ_to_bed_num(XYZ):
    # 计算marker位置
    x = XYZ[0]
    y = XYZ[1]
    z = XYZ[2]

    # x方向 plant and fruit 0.1 error，肯定够了
    if(3.9 < x < 4.1):
        bed_num_x_bias = 0
    elif(9.9 < x < 10.1):
        bed_num_x_bias = 9
    elif(15.9 < x < 16.1):
        bed_num_x_bias = 18
    else:
        bed_num_x_bias = 100
        print("invalid plant/fruit x")
    
    # y方向 plant and fruit 0.5 error，应该够了
    if(4 < y < 8):
        bed_num_y_bias = 0
    elif(11.5 < y < 15.5):
        bed_num_y_bias = 3
    elif(19 < y < 23):
        bed_num_y_bias = 6
    else:
        bed_num_y_bias = 100
        print("invalid plant/fruit y")
    
    # z方向 plant and fruit 0.5 error，应该够了
    if(0.6 < z < 1.6):
        bed_num_z_bias = 0
    elif(3.3 < z < 4.5):
        bed_num_z_bias = 1
    elif(6.2 < z < 7.2):
        bed_num_z_bias = 2
    else:
        bed_num_z_bias = 100
        print("invalid plant/fruit z")
    bed_num = bed_num_x_bias + bed_num_y_bias + bed_num_z_bias + 1
    return bed_num


    # 设置marker尺寸
    marker.scale.x = 0.5  # 可以根据需要调整
    marker.scale.y = 0.5
    marker.scale.z = 0.5

    # 设置marker颜色和透明度
    marker.color.r = 0.0  # 红色
    marker.color.g = 1.0  # 绿色
    marker.color.b = 0.0  # 蓝色
    marker.color.a = 0.8  # 不透明

    return marker

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
        self.plant_pub = rospy.Publisher("green_arr_ooad", MarkerArray, queue_size=10)
        self.fruit_pub = rospy.Publisher("fruit_arr_", MarkerArray, queue_size=10)
        self.green_arr_ = MarkerArray()
        self.fruit_arr_ = MarkerArray()

        self.green_dist = 1.0
        self.purple_dist = 0.17
        self.red_dist = 0.18
        self.yellow_dist = 0.13


    def add_plant_marker(self, plant_id, position, rpy_roll):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.99
        marker.scale.y = 0.99
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("plant pose isnan, return")
            return
        position[0] = round(position[0],0) # si she wu ru
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        # print("position[1]", position[1])
        # print("position[2]", position[2])
        marker.pose.orientation = Quaternion(0.0, 0.707, 0.0, 0.707)
        marker.id = plant_id

        rpy_roll = abs(rpy_roll)
        if(len(self.green_arr_.markers)>0):
            for old_marker in self.green_arr_.markers:
                dist = calc_marker_dist(old_marker, marker)
                if dist <= 1.0:
                    # print("duplicate green plant by 3D dist, IIR and return, rpy_roll is: ", rpy_roll)
                    old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                    old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                    old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                    return  # 如果距离小于或等于0.3，不添加Marker 并返回 (原MarkerArray)
        self.green_arr_.markers.append(marker)
        print("not duplicate, add one green")


    def add_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll):
        print("add_fruit_marker() called")
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
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
        for old_marker in self.fruit_arr_.markers:
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= self.yellow_dist and fruit_color == "yellow"):
                # print("duplicate fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                return
            if (dist <= self.red_dist and fruit_color == "red"):
                print("duplicate fruit by 3D dist, IIR and return")
                old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
                old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
                old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
                return
            if (dist <= self.purple_dist and fruit_color == "purple"):
                # print("duplicate fruit by 3D dist, IIR and return")
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
        self.plant_pub.publish(self.green_arr_)
        self.fruit_pub.publish(self.fruit_arr_)


