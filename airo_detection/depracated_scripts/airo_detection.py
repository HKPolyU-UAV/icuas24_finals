#!/usr/bin/env python
#write a python script that 
# 1) subscribes to the image topic
# 2) detectes the green area with a convex bounding box
# 3) in the convex boudning box, detect and publish the yellow fruit in the detected area with another bounding box
import sys, os
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Transform
import tf.transformations as tf_trans
from std_msgs.msg import String
from std_msgs.msg import Bool
from std_msgs.msg import Int32

from fruit_utils import *


class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_image_sub = Subscriber("/red/camera/color/image_raw", Image)
        self.depth_image_sub = Subscriber("/red/camera/depth/image_raw", Image)
        self.odom_sub = Subscriber("/red/odometry", Odometry)
        self.approx_time_synchronizer = ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub, self.odom_sub], queue_size=10, slop=0.0081)
        self.approx_time_synchronizer.registerCallback(self.DetectCallback)
        self.image_pub = rospy.Publisher("image_with_contours", Image, queue_size=10)
        self.depth_image_pub = rospy.Publisher("depth_image_with_contours", Image, queue_size=10)
        
        # 存储无人机pose
        # @todo 应该用 time_synchronizer 与 image 同步
        self.odom = Odometry()
        self.Rbody_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.intrinsic_matrix = np.array([[381.36, 0, 320.5], [0.0, 381.36, 240.5], [0, 0, 1]])

        # 创建一张空白的图片    
        self.img_height = 480
        self.img_width = 640
        self.cv_rgb_image_raw = np.zeros((self.img_height, self.img_width, 3), np.uint8)
        self.cv_depth_image_raw = np.zeros((self.img_height, self.img_width), np.float32)
        # self.cv_depth_image_raw = cv2.normalize(self.cv_depth_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 存储不同类型水果
        self.plant_fruit_database = PlantFruitDatabase()

        # Define range for green color PLANT
        self.lower_green = np.array([36, 25, 25])
        self.upper_green = np.array([70, 255, 255])
        # Define range for yellow color PEPPER
        self.lower_yellow = np.array([15, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        # Define range for red color TOMATO
        #self.lower_red = np.array([0, 70, 50])
        #self.upper_red = np.array([10, 100, 255])
        self.lower_red = np.array([0, 160, 50])
        self.upper_red = np.array([15, 255, 255])
        # Define range for purple color EGGPLANT
        self.lower_purple = np.array([130, 50, 70])
        self.upper_purple = np.array([160, 255, 255])

        # Define maximum and minimum contour area for a single green area
        self.max_area_green = 15000 
        self.min_area_green = 4000

        # Define maximum and minimum contour area for a single fruit
        #max_area_fruit = 1000
        self.max_area_fruit = 800
        # self.max_area_fruit = 1000
        self.min_area_fruit = 40

        self.yaw_threshold = 20
        self.roll_threshold = 25

        self.last_green_depth = 0

        # plant_beds related
        self.receive_bed_g = False
        self.bed_ids = [0]
        self.fruit_name = "no_fruit"
        self.fruit_color = "black"
        self.reached_endpoint_g = False
        self.has_pub_answer = False
        rospy.Subscriber("/red/plants_beds", String, self.plant_beds_callback)
        rospy.Subscriber("/reached_endpoint", Bool, self.reached_endpoint_callback)


    def reached_endpoint_callback(self, reached_endpoint_msg):
        self.reached_endpoint_g = reached_endpoint_msg.data

        if ((self.reached_endpoint_g == True) and (self.has_pub_answer == False)):
            final_fruit_num = len(self.plant_fruit_database.fruit_arr_.markers)
            fruit_count_pub = rospy.Publisher('fruit_count', Int32, queue_size=10, latch=True)
            fruit_count_pub.publish(final_fruit_num)
            print(f"Task finished, we have {final_fruit_num} {self.fruit_name}!!!")
            self.has_pub_answer = True

    def plant_beds_callback(self, plant_beds_msg):
        # global receive_bed_g
        # global ros_collion_free_sorted_cities_g

        print("self.bed_ids before callback: ", self.bed_ids)
        print("plant_beds_callback() called")
        if self.receive_bed_g==True:
            print("already received plant_beds, waypoint queue built")
            return
        else:
        # 将接收到的消息转换为字符串
            input_string = plant_beds_msg.data
            # 使用空格将字符串分割为数组
            elements = input_string.split()
            # 提取第一个元素为字符串
            self.fruit_name = elements[0]
            rospy.loginfo("Received fruit: %s", self.fruit_name)
            if(self.fruit_name == "Pepper"):
                self.fruit_color = "yellow"
            elif(self.fruit_name == "Eggplant"):
                self.fruit_color = "purple"
            elif(self.fruit_name == "Tomato"):
                self.fruit_color = "red"
            else:
                print("invalid fruit_name=====================")
            # 提取其他元素为整数数组
            self.bed_ids = [int(num) for num in elements[1:]]
            rospy.loginfo("Received bed numbers: %s", self.bed_ids)

        # 设置子坐标系ID
        self.receive_bed_g = True
        rospy.sleep(1)
        print("self.bed_ids after callback: ", self.bed_ids)
        rospy.sleep(1)
        print("receive_bed_g set as True")
    


    def findGreenContours(self, hsv_img):
        # Threshold the HSV image to get only green colors
        mask_green = cv2.inRange(hsv_img, self.lower_green, self.upper_green)

        ## Slice the green
        imask = mask_green>0
        green_area = np.zeros_like(hsv_img, np.uint8)
        green_area[imask] = hsv_img[imask]

        gray = cv2.cvtColor(green_area, cv2.COLOR_BGR2GRAY)
        # Find contours in the green mask
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


    def calcMeanDepthInGreenContour(self, green_contour):
        # Create a mask for the convex hull
        green_hull_func = cv2.convexHull(green_contour)
        green_area_mask_depth_func = np.zeros(self.cv_depth_image_raw.shape, dtype=np.uint8)
        # depth_mean_green = self.calc_mean_depth_in_green_contour()
        cv2.drawContours(green_area_mask_depth_func, [green_hull_func], -1, (255), thickness=cv2.FILLED)
        # 创建一个10x10的内核
        kernel = np.ones((11,11),np.uint8)
        # 使用erode函数
        green_area_mask_depth_func = cv2.erode(green_area_mask_depth_func, kernel, iterations = 1)
        # Bitwise-AND the mask and the original DEPTH image
        green_area_cropped_depth = cv2.bitwise_and(self.cv_depth_image_raw, self.cv_depth_image_raw, mask=green_area_mask_depth_func)
        # 计算平均深度
        depth_mean_green = cv2.mean(green_area_cropped_depth, mask=green_area_mask_depth_func)[0]
        return depth_mean_green

    def calcMeanDepthInFruitContour(self, fruit_contour):
        # Create a mask for the convex hull
        fruit_hull = cv2.convexHull(fruit_contour)
        fruit_area_mask_depth = np.zeros(self.cv_depth_image_raw.shape, dtype=np.uint8)
        cv2.drawContours(fruit_area_mask_depth, [fruit_hull], -1, (255), thickness=cv2.FILLED)
        # 创建一个10x10的内核
        # kernel = np.ones((11,11),np.uint8)
        # 使用erode函数
        # green_area_mask_depth = cv2.erode(green_area_mask_depth, kernel, iterations = 1)
        # Bitwise-AND the mask and the original DEPTH image
        fruit_area_cropped_depth = cv2.bitwise_and(self.cv_depth_image_raw, self.cv_depth_image_raw, mask=fruit_area_mask_depth)
        # 计算平均深度
        depth_mean_fruit = cv2.mean(fruit_area_cropped_depth, mask=fruit_area_mask_depth)[0]
        return depth_mean_fruit


    def uvd_to_cam_coor(self, u, v, depth):
        """
        计算物体在相机坐标系下的坐标
        参数:
        u,v: 物体在图像中的坐标 (u, v)
        depth: 物体的深度
        self.intrinsic_matrix: 相机的内参矩阵
        返回:
        物体在相机坐标系下的坐标 (X, Y, Z)
        """
        # 将图像坐标和深度合并为齐次坐标
        uv = np.array([u,v])
        uvd = np.append(uv,1)
        # print("uvd", uvd)
        # print("depth", depth)
        depth = depth + 0.14  # NOTE apply extrinsic 1
        u_v_depth = uvd * depth
        # print("u_v_depth", u_v_depth)

        # 计算物体在相机坐标系下的坐标  @为矩阵乘法 dot
        XYZcam = np.dot(np.linalg.inv(self.intrinsic_matrix),u_v_depth)

        return XYZcam

    def cam_coor_to_uv(self, XYZ_cam):
        # 如果Z坐标（深度）为0，那么该点在无穷远处，无法投影到图像上
        if XYZ_cam[2] == 0:
            raise ValueError("Point is at infinity.")

        # 将3D点从相机坐标系投影到图像平面
        point2D = np.dot(self.intrinsic_matrix, XYZ_cam)

        # 将齐次坐标转换为笛卡尔坐标
        u = point2D[0] / point2D[2]
        v = point2D[1] / point2D[2]

        return u, v
    
    def Tbody_cam(self, XYZ_cam):
        """
        将点从相机坐标系转换到body坐标系
        参数:
        XYZ_cam: 点在相机坐标系下的坐标 (X, Y, Z)
        rotation_matrix: 从相机坐标系到body坐标系的旋转矩阵
        [0, 0, 1
        -1, 0, 0
         0, -1, 0]
        返回:
        点在body坐标系下的坐标 (X, Y, Z)
        """
        # 将点从相机坐标系转换到body坐标系
        XYZ_body = np.dot(self.Rbody_cam, XYZ_cam)
        XYZ_body[0] = XYZ_body[0] # - 0.4  # NOTE apply extrinsic 2
        return XYZ_body
    
    def Tcam_body(self, XYZ_body):
        """
        将点从 body 坐标系 转换到 cam 坐标系
        参数:
        XYZ_body: 点在相机坐标系下的坐标 (X, Y, Z)
        rotation_matrix: 从相机坐标系到body坐标系的旋转矩阵
        [0, 0, 1
        -1, 0, 0
         0, -1, 0]
        返回:
        点在body坐标系下的坐标 (X, Y, Z)
        """
        # 将点从相机坐标系转换到body坐标系
        XYZ_cam = np.dot(np.linalg.inv(self.Rbody_cam), XYZ_body)
        return XYZ_cam
    
    def odom_to_transformation_matrix(self):
        # 提取位置和方向
        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation

        # 创建4x4变换矩阵
        transformation_matrix = np.eye(4)

        # 设置平移部分
        transformation_matrix[0, 3] = position.x
        transformation_matrix[1, 3] = position.y
        transformation_matrix[2, 3] = position.z

        # 设置旋转部分
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = tf_trans.quaternion_matrix(quaternion)
        transformation_matrix[:3, :3] = rotation_matrix[:3, :3]

        return transformation_matrix

    def Tworld_body(self, XYZ_body):
        """
        将点从body系转换到world坐标系
        参数:
        point_in_camera_frame: 点在相机坐标系下的坐标 (X, Y, Z)
        transform_matrix: 从 self.odom 中获得

        返回:
        点在world坐标系下的坐标 (X, Y, Z)
        """
        # 从 body coor to world coor
        # 从odometry消息中获取旋转矩阵和平移向量
        # 将点转换为齐次坐标
        transformation_matrix = self.odom_to_transformation_matrix()
        XYZ1_body = np.append(XYZ_body, 1)

        # 使用变换矩阵进行坐标变换
        point_in_world_coords_hom = np.dot(transformation_matrix, XYZ1_body)

        # 将结果转换回非齐次坐标
        XYZ_world = point_in_world_coords_hom[:3] / point_in_world_coords_hom[3]
        return XYZ_world
    
    def Tbody_world(self, XYZ_world):
        """
        将点从 world 系转换到 body 坐标系
        参数:
        XYZ_world: 点在世界坐标系下的坐标 (X, Y, Z)
        transform_matrix: 从 self.odom 中获得

        返回:
        点在world坐标系下的坐标 (X, Y, Z)
        """
        # 从 body coor to world coor
        # 从odometry消息中获取旋转矩阵和平移向量
        # 将点转换为齐次坐标
        transformation_matrix = self.odom_to_transformation_matrix()
        XYZ1_world = np.append(XYZ_world, 1)
        # 使用变换矩阵进行坐标变换
        point_in_body_coords_hom = np.dot(np.linalg.inv(transformation_matrix), XYZ1_world)

        # 将结果转换回非齐次坐标
        XYZ_body = point_in_body_coords_hom[:3] / point_in_body_coords_hom[3]
        return XYZ_body

    def uv_to_world(self, u, v, depth):
        """
        计算物体在相机坐标系下的坐标

        参数:
        u, v: 物体在图像中的坐标 (u, v)
        depth: 物体的深度
        intrinsic_matrix: 相机的内参矩阵

        返回:
        水果在世界坐标系下的坐标 (X, Y, Z)
        """
        XYZ_cam = self.uvd_to_cam_coor(u,v,depth)
        XYZ_body = self.Tbody_cam(XYZ_cam)
        XYZ_world = self.Tworld_body(XYZ_body)
        # print("XYZ_world is", XYZ_world)

        return XYZ_world
    
    def world_to_uv(self, XYZ_world_marker):
        # 将3D点从世界坐标系转换到相机坐标系
        XYZ_world = [0,0,0]
        XYZ_world[0] = XYZ_world_marker.pose.position.x
        XYZ_world[1] = XYZ_world_marker.pose.position.y
        XYZ_world[2] = XYZ_world_marker.pose.position.z
        XYZ_body = self.Tbody_world(XYZ_world)            
        # 将3D点从相机坐标系投影到图像平面
        XYZ_cam = self.Tcam_body(XYZ_body)
        u,v = self.cam_coor_to_uv(XYZ_cam)
        if(abs(XYZ_body[0]+XYZ_body[1]+XYZ_body[2])>5):
            # 如果到body系距离太远，让u,v 不能接受
            u = -1000
            v = -1000
        return u, v
    
    def fruits_in_FoV(self):
        fruits_in_FoV_arr = MarkerArray()
        for fruit_marker in self.plant_fruit_database.fruit_arr_.markers:
            u,v = self.world_to_uv(fruit_marker)
            if (100 < u < 540) and (0 < v < 480):
                fruits_in_FoV_arr.markers.append(fruit_marker)
        if len(fruits_in_FoV_arr.markers)==0:
            print("no fruit in FoV")
            return fruits_in_FoV_arr
        else:
            return fruits_in_FoV_arr

    def DetectCallback(self, rgb_msg, depth_msg, odom_msg):
        # dict_pepper = {1:0,2:0,3:0,4:0,5:4,6:0,7:3,8:0,9:0,\
        #      10:4,11:0,12:0,13:3,14:4,15:0,16:1,17:0,18:4,\
        #      19:0,20:0,21:4,22:3,23:1,24:3,25:0,26:0,27:1}
        # dict_tomato = {1:0,2:0,3:1,4:0,5:3,6:0,7:3,8:0,9:0,\
        #      10:0,11:3,12:0,13:1,14:3,15:0,16:3,17:0,18:0,\
        #      19:0,20:1,21:2,22:1,23:1,24:0,25:2,26:0,27:3}
        # dict_eggplant = {1:0,2:0,3:1,4:3,5:0,6:0,7:3,8:0,9:0,\
        #      10:0,11:0,12:1,13:2,14:2,15:0,16:0,17:3,18:4,\
        #      19:0,20:3,21:0,22:3,23:3,24:0,25:1,26:0,27:0}

        try:
            self.cv_rgb_image_raw = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        try:
            self.cv_depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except CvBridgeError as e:
            print(e)
            return
        # 计算平均深度
        # average_depth = np.nanmean(self.cv_depth_image_raw)
        # print('average depth for this depth image is: ', average_depth)

        self.odom = odom_msg
    
        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.cv_rgb_image_raw, cv2.COLOR_BGR2HSV)

        # Draw the largest 3 contours on the original image
        green_contours = self.findGreenContours(hsv)
        contour_image = self.cv_rgb_image_raw.copy()
        cv2.drawContours(contour_image, green_contours, -1, (255, 255, 0), 2)

        for cnt in sorted(green_contours, key=cv2.contourArea, reverse=True)[:3]:
            
            # Calculate the area of the contour
            area = cv2.contourArea(cnt)
            # Calculate the convex hull for the current contour
            green_hull = cv2.convexHull(cnt)
            # 计算mask部分的像素平均值
            x_mean_green = np.mean(green_hull[:,:,0])
            y_mean_green = np.mean(green_hull[:,:,1])
            # If the area is smaller than the maximum area for a single green area
            # only for plant at the center of iamge(distort)
            orientation_ywy = self.odom.pose.pose.orientation
            quaternion_ywy = [orientation_ywy.x, orientation_ywy.y, orientation_ywy.z, orientation_ywy.w]
            rpy_ywy = tf_trans.euler_from_quaternion(quaternion_ywy)
            rpy_ywy = [i * 57.29 for i in rpy_ywy]

            # print(f'area is: {area}')
            if self.min_area_green < area < self.max_area_green and 270 < x_mean_green < 370 and 10 < y_mean_green < 450:
            # if self.min_area_green < area < self.max_area_green:
                # print(f'area is: {area}')
                
                # Draw the convex hull on the original image
                cv2.drawContours(self.cv_rgb_image_raw, [green_hull], -1, (0, 255, 0), 3)
                
                cv2.drawContours(self.cv_depth_image_raw, [green_hull], -1, (10,10,10), 3)

                # print(f'\nmean x of green_area_depth_mask is: {x_mean_green}')
                # print(f'mean y of green_area_depth_mask is: {y_mean_green}')

                # CALCULATE GREEN DEPTH BY FUNCTION 
                depth_mean_green = self.calcMeanDepthInGreenContour(cnt)
                # depth_mean_green_3 = self.calcMeanDepthInGreenContour_copilot(cnt)
                # print(f"depth_mean_green is {depth_mean_green},======= ")

                if(abs(rpy_ywy[2]) <self.yaw_threshold):
                    depth_mean_green = depth_mean_green + 0.11
                elif(abs(abs(rpy_ywy[2])-180) <self.yaw_threshold):
                    depth_mean_green = depth_mean_green + 0.11
                else:
                    print("yaw is ", rpy_ywy[2])
                    print("NOT good yaw, return")
                    return

                # print("mean depth of green area is: ", depth_mean_green)
                len_fruit_arr = len(self.plant_fruit_database.fruit_arr_.markers)
                print(f"\nthe detected {self.fruit_name} number is: ============================================== {len_fruit_arr}")
                # real_pepper_num = dict_pepper[int(self.bed_ids[0])] + dict_pepper[int(self.bed_ids[1])] + dict_pepper[int(self.bed_ids[2])] + dict_pepper[int(self.bed_ids[3])] + dict_pepper[int(self.bed_ids[4])] + dict_pepper[int(self.bed_ids[5])] 
                # real_tomato_num = dict_tomato[int(self.bed_ids[0])] + dict_tomato[int(self.bed_ids[1])] + dict_tomato[int(self.bed_ids[2])] + dict_tomato[int(self.bed_ids[3])] + dict_tomato[int(self.bed_ids[4])] + dict_tomato[int(self.bed_ids[5])] 
                # real_eggplant_num = dict_eggplant[int(self.bed_ids[0])] + dict_eggplant[int(self.bed_ids[1])] + dict_eggplant[int(self.bed_ids[2])] + dict_eggplant[int(self.bed_ids[3])] + dict_eggplant[int(self.bed_ids[4])] + dict_eggplant[int(self.bed_ids[5])] 
                # print("the real number is: ================================ pepper:%d"%(real_pepper_num)+"   tomato:%d"%(real_tomato_num)+"   eggplant:%d"%(real_eggplant_num))

                XYZ_green = self.uv_to_world(x_mean_green, y_mean_green, depth_mean_green)
                # print("depth_mean_green is : ", depth_mean_green)
                # print("XYZ_green", XYZ_green)
                curr_plant_id = len(self.plant_fruit_database.green_arr_.markers) + 100
                plant_bed_num = XYZ_to_bed_num(XYZ_green)
                if(plant_bed_num in self.bed_ids):
                    if(abs(rpy_ywy[0]) <self.roll_threshold):
                        # print(f"roll is : {rpy_ywy[0]}, okay")
                        pass
                    else:
                        print("roll is =================================: ", rpy_ywy[0])
                        print("==================================NOT good roll, return")
                        return
                    print("in bed_ids, id is: ", plant_bed_num)
                    self.plant_fruit_database.add_plant_marker(curr_plant_id, XYZ_green, abs(rpy_ywy[0]))
                else:
                    print("NOT in bed_ids, id is: ", plant_bed_num)
                    return
                
                # Create a mask for yellow color inside the green area
                green_area_mask_rgb = np.zeros(self.cv_rgb_image_raw.shape[:2], np.uint8)
                cv2.drawContours(green_area_mask_rgb, [green_hull], -1, (255), thickness=cv2.FILLED)
                # Bitwise-AND the mask and the original image RGB->HSV
                green_area_cropped_rgb = cv2.bitwise_and(hsv, hsv, mask=green_area_mask_rgb)

                if(self.fruit_color == "yellow"):
                    # Create a mask for red color inside the green area
                    mask_yellow_in_green = cv2.inRange(green_area_cropped_rgb, self.lower_yellow, self.upper_yellow)
                    contours_yellow_pre, _ = cv2.findContours(mask_yellow_in_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # contours_yellow_pre, _ = cv2.findContours(dilated.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # NOTE 对 yellow_pepper 作专门处理
                    # 因为 yellow_pepper 有时候会两个水果贴在一起，而 red 和 purple目前没遇到这种情况
                    if(len(contours_yellow_pre) >= 2):
                        dist_transform = cv2.distanceTransform(mask_yellow_in_green, cv2.DIST_L2, 5)
                        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
                        _, dist_transform_threshold = cv2.threshold(dist_transform, 0.52*dist_transform.max(), 255, cv2.THRESH_BINARY)
                        # Find contours in the yellow mask
                        contours_yellow, _ = cv2.findContours(dist_transform_threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    else:
                        print("no more than 1 contours====================")
                        contours_yellow = contours_yellow_pre
                    # init contours in other two colors as empty
                    contours_red = []
                    contours_purple = []
                elif(self.fruit_color == "red"):
                    # Create a mask for red color inside the green area
                    mask_red_in_green = cv2.inRange(green_area_cropped_rgb, self.lower_red, self.upper_red)
                    # Find contours in the red mask
                    contours_red, _ = cv2.findContours(mask_red_in_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_yellow = []
                    contours_purple = []
                elif(self.fruit_color == "purple"):
                    # Create a mask for purple color inside the green area
                    mask_purple_in_green = cv2.inRange(green_area_cropped_rgb, self.lower_purple, self.upper_purple)
                    # Find contours in the purple mask
                    contours_purple, _ = cv2.findContours(mask_purple_in_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_red = []
                    contours_yellow = []
                else:
                    print("invalid fruit color======, return")
                    return


                # Draw the convex contours of the yellow areas on the original image
                
                # print("\n\n\nlength of contours_yellow is: ", len(contours_yellow))
                for contour_yellow in contours_yellow:
                    # Calculate the area of the contour
                    area = cv2.contourArea(contour_yellow)
                    # If the area is smaller than the maximum area for a single fruit                    
                    cv2.drawContours(self.cv_rgb_image_raw, [contour_yellow], -1, (10, 10, 10), -1)

                    hull_yellow = cv2.convexHull(contour_yellow)
                    cv2.drawContours(self.cv_rgb_image_raw, [hull_yellow], -1, (0, 0, 255), 2)

                    # 找到轮廓的边界框
                    x, y, w, h = cv2.boundingRect(contour_yellow)

                    if self.min_area_fruit < area < self.max_area_fruit:  
                        # 使用cv2.circle()函数画圆
                        x_mean_yellow = x + w/2
                        y_mean_yellow = y + h/2
                        cv2.circle(self.cv_rgb_image_raw, (int(x_mean_yellow), int(y_mean_yellow)), 10, (255,0,0), 4)
                        
                        len_fruit_arr = len(self.plant_fruit_database.fruit_arr_.markers)
                        # 初始化
                        if (len_fruit_arr==0):
                            # print("the fruit arr is empty")
                            depth_mean_yellow = depth_mean_green
                            XYZ_yellow = self.uv_to_world(x_mean_yellow, y_mean_yellow, depth_mean_yellow)
                            curr_yellow_id = len(self.plant_fruit_database.fruit_arr_.markers)
                            self.plant_fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))
                        
                        # depth_mean_yellow = depth_mean_green # NOTE 难道水果的深度不受影响？ actually, fruit depth is 0.15 smaller than plant depth
                        depth_mean_yellow = self.calcMeanDepthInFruitContour(contour_yellow)
                        print(f"depth_mean_yellow: {depth_mean_yellow},======== depth_mean_yellow_2(by FUNC each fruit): {depth_mean_yellow} ")
                        XYZ_yellow = self.uv_to_world(x_mean_yellow, y_mean_yellow, depth_mean_yellow)
                        curr_yellow_id = len(self.plant_fruit_database.fruit_arr_.markers)
                        # print("XYZ_yellow", XYZ_yellow)
                        self.plant_fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))

                for contour_purple in contours_purple:
                    # Calculate the area of the contour
                    area = cv2.contourArea(contour_purple)
                    # If the area is smaller than the maximum area for a single fruit                    
                    cv2.drawContours(self.cv_rgb_image_raw, [contour_purple], -1, (10, 10, 10), -1)

                    hull_purple = cv2.convexHull(contour_purple)
                    cv2.drawContours(self.cv_rgb_image_raw, [hull_purple], -1, (0, 0, 255), 2)

                    # 找到轮廓的边界框
                    x, y, w, h = cv2.boundingRect(contour_purple)

                    if self.min_area_fruit < area < self.max_area_fruit:  
                        # 使用cv2.circle()函数画圆
                        x_mean_purple = x + w/2
                        y_mean_purple = y + h/2
                        cv2.circle(self.cv_rgb_image_raw, (int(x_mean_purple), int(y_mean_purple)), 10, (255,0,0), 4)
                        
                        len_fruit_arr = len(self.plant_fruit_database.fruit_arr_.markers)
                        # print("the fruit number is: ", len_fruit_arr)
                        # 初始化
                        if (len_fruit_arr==0):
                            # print("the fruit arr is empty")
                            depth_mean_purple = self.calcMeanDepthInFruitContour(contour_purple)
                            XYZ_purple = self.uv_to_world(x_mean_purple, y_mean_purple, depth_mean_purple)
                            curr_purple_id = len(self.plant_fruit_database.fruit_arr_.markers)
                            self.plant_fruit_database.add_fruit_marker('purple', curr_purple_id, XYZ_purple, abs(rpy_ywy[0]))
                        
                        # depth_mean_purple = self.calcMeanDepthInFruitContour(contour_purple)
                        depth_mean_purple = depth_mean_green #NOTE sometimes, eggplant depth is wrong
                        XYZ_purple = self.uv_to_world(x_mean_purple, y_mean_purple, depth_mean_purple)
                        curr_purple_id = len(self.plant_fruit_database.fruit_arr_.markers)
                        # print("XYZ_purple", XYZ_purple)
                        self.plant_fruit_database.add_fruit_marker('purple', curr_purple_id, XYZ_purple, abs(rpy_ywy[0]))
                   
                for contour_red in contours_red:
                    # Calculate the area of the contour
                    area = cv2.contourArea(contour_red)
                    # If the area is smaller than the maximum area for a single fruit                    
                    cv2.drawContours(self.cv_rgb_image_raw, [contour_red], -1, (10, 10, 10), -1)

                    hull_red = cv2.convexHull(contour_red)
                    cv2.drawContours(self.cv_rgb_image_raw, [hull_red], -1, (0, 0, 255), 2)

                    # 找到轮廓的边界框
                    x, y, w, h = cv2.boundingRect(contour_red)

                    if self.min_area_fruit < area < self.max_area_fruit:  
                        # 使用cv2.circle()函数画圆
                        x_mean_red = x + w/2
                        y_mean_red = y + h/2
                        cv2.circle(self.cv_rgb_image_raw, (int(x_mean_red), int(y_mean_red)), 10, (255,0,0), 4)
                        
                        len_fruit_arr = len(self.plant_fruit_database.fruit_arr_.markers)
                        # print("the fruit number is: ", len_fruit_arr)
                        # 初始化
                        if (len_fruit_arr==0):
                            print("the fruit arr is empty")
                            depth_mean_red = self.calcMeanDepthInFruitContour(contour_red)
                            XYZ_red = self.uv_to_world(x_mean_red, y_mean_red, depth_mean_red)
                            curr_red_id = len(self.plant_fruit_database.fruit_arr_.markers)
                            self.plant_fruit_database.add_fruit_marker('red', curr_red_id, XYZ_red, abs(rpy_ywy[0]))
                        
                        # depth_mean_red = self.calcMeanDepthInFruitContour(contour_red)
                        depth_mean_red = depth_mean_green #NOTE still, we find green_dpeth is more robust than red_depth, but NOT suitable for yellow
                        XYZ_red = self.uv_to_world(x_mean_red, y_mean_red, depth_mean_red)
                        curr_red_id = len(self.plant_fruit_database.fruit_arr_.markers)
                        # print("XYZ_red", XYZ_red)
                        self.plant_fruit_database.add_fruit_marker('red', curr_red_id, XYZ_red, abs(rpy_ywy[0]))

                
        # Convert the image to a message
        image_msg = self.bridge.cv2_to_imgmsg(self.cv_rgb_image_raw, "bgr8")
        depth_image_msg = self.bridge.cv2_to_imgmsg(self.cv_depth_image_raw, "passthrough")

        # Publish the image
        self.image_pub.publish(image_msg)
        self.depth_image_pub.publish(depth_image_msg)
        self.plant_fruit_database.publish_markers()


def main():
    rospy.init_node('image_processor', anonymous=True)
    ic = ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
