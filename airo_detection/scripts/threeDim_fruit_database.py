import rospy
import numpy as np
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int32

# BAG 1 fruit count: 18 yellow, 3 larger red, 5 smaller red
# BAG 2 fruit count: 18 yellow, 3 larger red, 5 smaller red
# BAG 3 fruit count: 18 yellow, 3 larger red, 5 smaller red
# BAG 4 fruit count: 23 yellow, 7 larger red, 10 smaller red

def calc_marker_dist(marker1, marker2):
    dx = marker1.pose.position.x - marker2.pose.position.x
    dy = marker1.pose.position.y - marker2.pose.position.y
    dz = marker1.pose.position.z - marker2.pose.position.z
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    if math.isnan(dist):
        print("dist is nan")
        return 0
    return dist



class PlantFruitDatabase:
    def __init__(self):
        self.red_fruit_pub = rospy.Publisher("/red_fruit_arr_", MarkerArray, queue_size=10)
        self.red_fruit_count_pub = rospy.Publisher('/red_fruit_count', Int32, queue_size=10, latch=True)
        self.red_fruit_arr_ = MarkerArray()
        self.red_fruit_list_ =[]
        self.yellow_fruit_pub = rospy.Publisher("/yellow_fruit_arr_", MarkerArray, queue_size=10)
        self.yellow_fruit_count_pub = rospy.Publisher('/yellow_fruit_count', Int32, queue_size=10, latch=True)
        self.yellow_fruit_arr_ = MarkerArray()
        self.real_yellow_fruit_arr_ = []
        self.real_red_fruit_arr_ = []

        self.red_dist = 1.1
        self.red_max_prob = 0.3
        self.red_2nd_max_prob = 0.3
        self.red_3rd_max_prob = 0.3
        self.red_prob_mean = 0.1

        self.yellow_dist = 1.0
        self.yellow_max_prob = 0.3
        self.yellow_2nd_max_prob = 0.3
        self.yellow_3rd_max_prob = 0.3
        self.yellow_prob_mean = 0.1

    def add_red_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll, two_d_size):
        print(f"add_red_fruit_marker() called, the pose is\n{position}")
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        fruit_size = two_d_size/10
        marker.scale.x = fruit_size
        marker.scale.y = fruit_size
        marker.scale.z = fruit_size
        marker.color.a = 0.1
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("fruit pose isnan, return")
            return

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.id = fruit_id

        rpy_roll = abs(rpy_roll)
        self.real_red_fruit_arr_.clear()
        this_id = 0
        half_count = 0
        closest_old_marker_id = 150
        closest_old_marker_dist = 10

        new_red_prob_sum = 0
        for i in range(0,len(self.red_fruit_arr_.markers)):
            print(f"the len(self.red_fruit_arr_.markers is {len(self.red_fruit_arr_.markers)})")
            old_marker = self.red_fruit_arr_.markers[i]
            new_red_prob_sum = new_red_prob_sum + old_marker.color.a
            # if(old_marker.color.a > self.red_max_prob):
            #     self.red_max_prob = old_marker.color.a
            # if(old_marker.color.a > self.red_2nd_max_prob and old_marker.color.a < self.red_max_prob):
            #     self.red_2nd_max_prob = old_marker.color.a
            # if(old_marker.color.a > self.red_3rd_max_prob and old_marker.color.a < self.red_2nd_max_prob):
            #     self.red_3rd_max_prob = old_marker.color.a
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= closest_old_marker_dist):
                closest_old_marker_dist = dist
                closest_old_marker_id = i
            if(self.red_prob_mean < 1.7):
                if(old_marker.color.a > min(0.4, (self.red_prob_mean))):
                    self.real_red_fruit_arr_.append(1)
                elif((min(0.4, (self.red_prob_mean))) >= old_marker.color.a >= 0.3):
                    half_count = half_count+0.5
                elif(old_marker.color.a < 0.3):
                    print(f"real_red_fruit_arr_: {self.real_red_fruit_arr_}")
            else:
                if(old_marker.color.a > max(0.4, (self.red_prob_mean))):
                    self.real_red_fruit_arr_.append(1)
                elif((min(0.4, (self.red_prob_mean))) >= old_marker.color.a >= 0.3):
                    half_count = half_count+0.5
                elif(old_marker.color.a < 0.3):
                    print(f"real_red_fruit_arr_: {self.real_red_fruit_arr_}")
        # endfor
        new_red_prob_mean = new_red_prob_sum/(len(self.red_fruit_arr_.markers) + 0.3)
        self.red_prob_mean = new_red_prob_mean
        if(closest_old_marker_dist < self.red_dist):
            print("the closest fruit is the {closest_old_marker_id}th red with dist = {closest_old_marker_dist}")
            old_marker = self.red_fruit_arr_.markers[closest_old_marker_id]
            print("duplicate red_fruit by 3D dist, IIR and return")
            old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
            old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
            old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
            old_marker.color.a = old_marker.color.a + 0.1
            old_marker.scale.x = (3*old_marker.scale.x + fruit_size)/4
            old_marker.scale.y = (3*old_marker.scale.y + fruit_size)/4
            old_marker.scale.z = (3*old_marker.scale.z + fruit_size)/4
            # self.red_fruit_count_pub.publish(i+1)
            print(f"red_id by IIR is: {closest_old_marker_id+1}, start from 1, NOT 0")
            this_id = closest_old_marker_id+1
        # if it's a new red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        if(this_id == 0):
            self.red_fruit_arr_.markers.append(marker)
            # self.red_fruit_list_.append(marker)
        # new_red_id = len(self.red_fruit_arr_.markers)
        self.red_fruit_count_pub.publish(len(self.real_red_fruit_arr_)+ int(half_count))
        if(this_id > len(self.real_red_fruit_arr_) or this_id==0):
            this_id = len(self.real_red_fruit_arr_)
        return this_id

    def add_yellow_fruit_marker(self, fruit_color, fruit_id, position, rpy_roll, two_d_size):
        print("add_yellow_fruit_marker() called")
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        fruit_size = two_d_size/10
        marker.scale.x = fruit_size
        marker.scale.y = fruit_size
        marker.scale.z = fruit_size
        marker.color.a = 0.1
        if(np.isnan(position[0]) or np.isnan(position[1]) or np.isnan(position[2])):
            print("fruit pose isnan, return")
            return

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation = Quaternion(0,0,0,1)
        marker.id = fruit_id

        rpy_roll = abs(rpy_roll)

        self.real_yellow_fruit_arr_.clear()
        this_id = 0
        half_count = 0
        closest_old_marker_id = 50
        closest_old_marker_dist = 10
        
        new_yellow_prob_sum = 0
        for i in range(0,len(self.yellow_fruit_arr_.markers)):
            old_marker = self.yellow_fruit_arr_.markers[i]
            new_yellow_prob_sum = new_yellow_prob_sum + old_marker.color.a
            # if(old_marker.color.a > self.yellow_max_prob):
            #     self.yellow_max_prob = old_marker.color.a
            # if(old_marker.color.a > self.yellow_2nd_max_prob and old_marker.color.a < self.yellow_max_prob):
            #     self.yellow_2nd_max_prob = old_marker.color.a
            # if(old_marker.color.a > self.yellow_3rd_max_prob and old_marker.color.a < self.yellow_2nd_max_prob):
            #     self.yellow_3rd_max_prob = old_marker.color.a
            dist = calc_marker_dist(old_marker, marker)
            if (dist <= closest_old_marker_dist):
                closest_old_marker_dist = dist
                closest_old_marker_id = i

            if(self.yellow_prob_mean < 1.7):
                if(old_marker.color.a > (min(0.4, (self.yellow_prob_mean)*0.8))):
                    self.real_yellow_fruit_arr_.append(1)
                elif((min(0.4, (self.yellow_prob_mean)*0.8)) >= old_marker.color.a >= 0.3):
                    half_count = half_count+0.5
                elif(old_marker.color.a < 0.3):
                    print(f"real_yellow_fruit_arr_: {self.real_yellow_fruit_arr_}")
            else:
                if(old_marker.color.a > (max(0.4, (self.yellow_prob_mean)*0.8))):
                    self.real_yellow_fruit_arr_.append(1)
                elif((max(0.4, (self.yellow_prob_mean)*0.8)) >= old_marker.color.a >= 0.3):
                    half_count = half_count+0.5
                elif(old_marker.color.a < 0.3):
                    print(f"real_yellow_fruit_arr_: {self.real_yellow_fruit_arr_}")

        # endfor
        new_yellow_prob_mean = new_yellow_prob_sum/(len(self.yellow_fruit_arr_.markers) + 0.3)
        self.yellow_prob_mean = new_yellow_prob_mean
                
        if(closest_old_marker_dist < self.yellow_dist):
            print("the closest fruit is the {closest_old_marker_id}th yellow with dist = {closest_old_marker_dist}")
            old_marker = self.yellow_fruit_arr_.markers[closest_old_marker_id]
            print("duplicate yellow_fruit by 3D dist, IIR and return")
            old_marker.pose.position.x = (rpy_roll*old_marker.pose.position.x + marker.pose.position.x)/(rpy_roll+1)
            old_marker.pose.position.y = (rpy_roll*old_marker.pose.position.y + marker.pose.position.y)/(rpy_roll+1)
            old_marker.pose.position.z = (rpy_roll*old_marker.pose.position.z + marker.pose.position.z)/(rpy_roll+1)
            old_marker.color.a = old_marker.color.a + 0.1
            old_marker.scale.x = (3*old_marker.scale.x + fruit_size)/4
            old_marker.scale.y = (3*old_marker.scale.y + fruit_size)/4
            old_marker.scale.z = (3*old_marker.scale.z + fruit_size)/4
            # self.yellow_fruit_count_pub.publish(i+1)
            print(f"yellow_id by IIR is: {closest_old_marker_id+1}, start from 1, NOT 0")
            this_id = closest_old_marker_id+1
        # if it's a new yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if(this_id == 0):
            self.yellow_fruit_arr_.markers.append(marker)
        # new_yellow_id = len(self.yellow_fruit_arr_.markers)
        self.yellow_fruit_count_pub.publish(len(self.real_yellow_fruit_arr_)+int(half_count))
        # print(f"new_yellow_id is: {new_yellow_id}")
        if(this_id > len(self.real_yellow_fruit_arr_) or this_id==0):
            this_id = len(self.real_yellow_fruit_arr_)
        return this_id




    def publish_markers(self):
        # print("publish from class===================================================")
        self.red_fruit_pub.publish(self.red_fruit_arr_)
        self.yellow_fruit_pub.publish(self.yellow_fruit_arr_)
