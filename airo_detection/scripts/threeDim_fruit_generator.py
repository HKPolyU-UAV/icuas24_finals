import rospy
import cv2
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import message_filters
import statistics
import matplotlib.pyplot as plt
import io
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import tf.transformations as tf_trans
from scipy.spatial import distance
import queue
from ultralytics import YOLO
import time



import sys, os
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)

from threeDim_fruit_database import *
from twoDim_fruit_detector_ywy import TwoDFruitDetector
# from twoDim_fruit_detector import TwoDFruitDetector
from transform_utils import *
from yolo_detect import *
import tf

def quaternion_to_rotation_matrix(quat):
    return tf.transformations.quaternion_matrix(quat)[:3, :3]


def odom_msg_to_rpy(odom_msg):
    orientation = odom_msg.pose.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    rpy_rad = tf_trans.euler_from_quaternion(quaternion)
    # rpy_deg = [i * 57.29 for i in rpy_rad]
    rpy_deg = np.rad2deg(rpy_rad)
    return rpy_deg


class LidarReprojector:
    def __init__(self):
        # Camera intrinsics and T_cam_lidar
        self.queue_size = 2
        self.knn_mode_percent = 80
        # self.knn_dist = 2
        self.bridge = CvBridge()
        self.fx = 624.325
        self.fy = 624.439
        self.cx = 320.585
        self.cy = 219.946
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)
        self.dist_coeffs = np.zeros((5, 1), np.float32)
        self.R_cam_lidar_z_neg90 =  np.array([[0,1,0],
                                [-1, 0, 0],
                                [0,0.,1] ])
                                
        # approximate
        self.R_cam_lidar_y_neg104 =  np.array([[-0.2419219,  0.0000000, -0.9702957],
                                        [0.0000000,  1.0000000,  0.0000000],
                                        [0.9702957,  0.0000000, -0.2419219] ])
        
        # approximate
        self.R_cam_lidar_y_neg102 =  np.array([[ -0.2079117,  0.0000000, -0.9781476],
                                            [0.0000000,  1.0000000,  0.0000000],
                                            [0.9781476,  0.0000000, -0.2079117 ]])
        
        self.R_cam_lidar_y_neg101point5 =  np.array([[ -0.1993679,  0.0000000, -0.9799247],
                                            [0.0000000,  1.0000000,  0.0000000],
                                            [0.9799247,  0.0000000, -0.1993679 ]])

        # x:180 y:-11.5
        self.R_imu_lidar = np.array([[ 0.9799247,  0.0000000, -0.1993679],
                                [-0.0000000, -1.0000000, -0.0000000],
                                [-0.1993679,  0.0000000, -0.9799247 ]])


        self.R_cam_lidar = np.dot(self.R_cam_lidar_z_neg90, self.R_cam_lidar_y_neg101point5)
        self.R_lidar_imu = np.linalg.inv(self.R_imu_lidar)
        self.R_cam_imu = np.dot(self.R_cam_lidar, self.R_lidar_imu)
        # self.R_cam_lidar = np.dot(np.array([[0,1,0], [-1, 0, 0], [0,0.,1]]), np.array([[-0.2419219,  0.0000000, -0.9702957], [0.0000000,  1.0000000,  0.0000000], [0.9702957,  0.0000000, -0.2419219]]))
        self.rotvec_cam_lidar_g, _ = cv2.Rodrigues(self.R_cam_lidar)
        self.transvec_cam_lidar_g = np.array([0.0,0.15,-0.1])
        self.transvec_lidar_imu_g = np.array([0.0,0.15,-0.25])
        self.transvec_cam_imu_g = np.array([-0.07,0.0,-0.15])
        self.lidar_projected_image_pub = rospy.Publisher('/fused_image_ooad', Image, queue_size=10)
        self.fruit_detections_pub = rospy.Publisher('/fruit_detections', Image, queue_size=10)

        self.fruit_database = PlantFruitDatabase()
        self.transform_utils = TransformUtils()
        self.twoD_fruit_detector = TwoDFruitDetector()
        self.yolo_model = YOLO("/root/lala_ws/src/icuas24_finals/airo_detection/scripts/detect_fruit_test/demo_data/best.pt")


        # ROS node and subscriber
        rospy.init_node('lidar_reprojector', anonymous=True)
        # rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)

        # Subscribe to both topics
        # self.sub_lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        self.sub_lidar = message_filters.Subscriber('/undistored_points', PointCloud2)
        self.sub_image = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        # self.sub_2dfruit = message_filters.Subscriber('two_d_fruit_keypoints', PointStamped)
        # self.sub_odom = message_filters.Subscriber('/kiss/odometry', Odometry)
        self.sub_odom = message_filters.Subscriber('/Odometry', Odometry)
        self.cam_fov_pub = rospy.Publisher('cam_fov_visualization', Marker, queue_size=10)
        self.quad_rviz_pub = rospy.Publisher('quadrotor', MarkerArray, queue_size=10)
        # Synchronize the topics with a slop of 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_lidar, self.sub_odom], 3000, 0.08)
        self.ts.registerCallback(self.callback)

        # Create queues for lidar_msg and odom_msg
        self.lidar_msg_queue = queue.Queue(maxsize=self.queue_size)
        self.odom_msg_queue = queue.Queue(maxsize=self.queue_size)
        self.image_msg_queue = queue.Queue(maxsize=10)
        self.frame_count = 0
        self.min_yellow_size = 10
        self.max_yellow_size = 10
        self.min_red_size = 10
        self.max_red_size = 10

        self.last_odom = Odometry()
        self.last_odom.pose.pose.position = Point(0, 0, 0)
        self.curr_odom = Odometry()
        self.path_length = 0.0
        self.roll = 0.0
        self.pitch = 0.0


        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.color_y = (0, 255, 255)  # BGR color for blue
        self.color_r = (0, 0, 255)  # BGR color for blue


    def callback(self, image_msg, lidar_msg, odom_msg):
        print(f"\n\n=============================the time we entered callback is :{time.time()}")
        time_gap = (lidar_msg.header.stamp - image_msg.header.stamp).to_sec()
        print(f"the time gap lidar_time-image_time is: ======================{time_gap:.3f}")
        print(f"good lidar and image match, airo detection here")
        print(f"we have poped all three msgs for processing, for fruit generation")
        
        rpy_deg = odom_msg_to_rpy(odom_msg)
        self.roll = rpy_deg[0]
        self.pitch = rpy_deg[1]
        print(f"rpy_deg is : {rpy_deg}")
        if(abs(self.roll) > 4.4):
        # if(abs(rpy_deg[0]) > 6.5):
            print(f"roll is {rpy_deg[0]}, too large, we NOT generate 3D fruit in this frame")
            print(f"return of this callback function")
            return
        if(abs(self.pitch) > 4.4):
        # if(abs(rpy_deg[0]) > 6.5):
            print(f"pitch is {rpy_deg[0]}, too large, we NOT generate 3D fruit in this frame")
            print(f"return of this callback function")
            return
        
        # print("we are inside the callback now")
        self.publish_camera_fov_marker(odom_msg)
        self.publish_quad_rviz(odom_msg)

        print(f"=======================the time add lidar_msg in queue and pub visualize markers :{time.time()}")

        

        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Reproject points
        points_arr = self.lidar_msg_to_xyz_array(lidar_msg)

        print(f"=============================================the time we turn ldiar msg to points_arr :{time.time()}")
        uvd_points = self.lidar_points_to_uvd(points_arr)  # YWY noly need uvd
        print(f"==========================================the time we called lidar_points_to_uvs :{time.time()}")
        
        # fruit_points = self.twoD_fruit_detector.detect_fruit(image)
        yolo_fruit_yellows, yolo_fruit_reds = yolo_detect(image, self.yolo_model)
        print(f"=================================================the time we called yolo :{time.time()}")
        
        image_fruit_detections = image.copy()
        if(len(self.fruit_database.red_fruit_arr_.markers)>0 or len(self.fruit_database.yellow_fruit_arr_.markers)>0):
            print(f"draw a new fruit_arr_ on a new image ===============================================")
            uvd_fruits = self.fruit_markers_to_uvd(odom_msg)
            image = self.colorize_fruits_uvd(uvd_fruits, image)

        image_with_lidar_points, image_mean_depth, middle_image_mean_depth = self.colorize_uvd_visualization(uvd_points, image)  #colored_image_points only for visualization, could delete
        if (middle_image_mean_depth<4.4):
            # @TODO 1 fit gaussian
            # @TODO 2 in find fruit_depth, find fruit depth with gaussian
            print(f"middle_image_mean_depth is {middle_image_mean_depth}, we should do gaussian fitting here")
            means = self.calculate_gaussian(uvd_points, odom_msg)  #
            print(f"the depth means are {means}")
        else:
            means = self.calculate_gaussian(uvd_points, odom_msg)  #
            print(f"image_mean_depth is {image_mean_depth}, no gaussian fitting")
        
        means0 = "depth_mean0: "+ str(means[0])        
        means1 = "depth_mean1: "+ str(means[1])
        cv2.putText(image_with_lidar_points, means0, (30,50), self.font, self.font_scale, self.color_y, thickness=2)
        cv2.putText(image_with_lidar_points, means1, (30,80), self.font, self.font_scale, self.color_y, thickness=2)

        self.add_yellow_fruits_from_yolo_detection(image_mean_depth, means, yolo_fruit_yellows, "yellow", uvd_points, image_with_lidar_points, odom_msg, image_fruit_detections)
        self.add_red_fruits_from_yolo_detection(image_mean_depth, means, yolo_fruit_reds, "red", uvd_points, image_with_lidar_points, odom_msg, image_fruit_detections)

        colored_image_msg = self.bridge.cv2_to_imgmsg(image_with_lidar_points, "bgr8")
        colored_image_msg.header = odom_msg.header
        self.lidar_projected_image_pub.publish(colored_image_msg)

        self.calculate_path_length_3d()
        self.last_odom = self.curr_odom
        self.curr_odom = odom_msg
        # image.clear()
        # uvd_points.clear()

    def calculate_path_length_3d(self):
        dx = self.curr_odom.pose.pose.position.x - self.last_odom.pose.pose.position.x
        dy = self.curr_odom.pose.pose.position.y - self.last_odom.pose.pose.position.y
        dz = self.curr_odom.pose.pose.position.z - self.last_odom.pose.pose.position.z
        self.path_length += (dx**2 + dy**2 + dz**2)**0.5
        return self.path_length

    # def fruit_reprojection(self, fruit_XYZ_world):
    #     fruit_XYZ_lidar = self.transform_utils.Tlidar_world(fruit_XYZ_world)
    #     return fruit_XYZ_lidar

    def add_yellow_fruits_from_yolo_detection(self, image_mean_depth, means, yolo_fruits, color, uvd_points, image_with_lidar_points, odom_msg, image_fruit_detections):
        for fruit_point in yolo_fruits:
            # print(f"rpy_deg is : {rpy_deg}")
            print(f"yellow fruit is at ({int(fruit_point.x)},{int(fruit_point.y)}, {int(fruit_point.z)})")
            # fruit_point = PointStamped()
            if(fruit_point.x<40 or fruit_point.x >600):
                return
            # fruit_point = two_d_fruit_keypoints_msg.point
            image_with_lidar_points = self.draw_bbx(image_with_lidar_points, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z))

            # fruit_point.z = fruit_point.z
            fruit_depth1, knn_points = self.find_yellow_fruit_depth(uvd_points, fruit_point)
            fruit_depth2 = self.find_yolo_yellow_depth(float(fruit_point.z))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",fruit_depth1)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",fruit_depth2)
            for i in range(0,len(knn_points)):
                # Draw the line
                p = knn_points
                cv2.line(image_with_lidar_points, (int(p[i][0]), int(p[i][1])), (int(fruit_point.x), int(fruit_point.y)), (0,255,255), 2)
            position1 = (int(fruit_point.x+20), int(fruit_point.y)-20) 
            position2 = (int(fruit_point.x+20), int(fruit_point.y)+20) 
            dpeth_str1 = str(fruit_depth1)
            cv2.putText(image_with_lidar_points, dpeth_str1, position1, self.font, self.font_scale, self.color_y, thickness=2)
            dpeth_str2 = str(fruit_depth2)
            cv2.putText(image_with_lidar_points, dpeth_str2, position2, self.font, self.font_scale, self.color_y, thickness=2)

            if(fruit_depth1 == None)or(fruit_depth1 == False):
                fruit_depth = fruit_depth1
                print('we have return here add_yellow_fruits_from_yolo_detection')
                return
            else:
                # fruit_depth1 = fruit_depth1
                fruit_depth = fruit_depth1 * 0.2 + fruit_depth2 * 0.8
          
            # XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
            
            print(f"has_valid_depth is {fruit_depth}")
            if(not fruit_depth):
                print(f"no valid depth")
                return
            elif(fruit_depth):
                print(f"has valid depth")
                if(image_mean_depth < 6):
                    depth_diff0 = abs(fruit_depth - means[0])
                    depth_diff1 = abs(fruit_depth - means[1])
                    if(depth_diff0 < depth_diff1):
                        fruit_depth = fruit_depth*0.2 + means[0]*0.8
                    else:
                        fruit_depth = fruit_depth*0.2 + means[1]*0.8
                XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color_g = (0, 255, 0)  # BGR color for blue
                color_r = (0, 0, 0)  # BGR color for blue
                # Position for the text
                position = (int(fruit_point.x), int(fruit_point.y))  # You can adjust this according to your needs

                # if(fruit_depth > 3 and fruit_depth < 9 and XYZ_yellow[0] > 2 and XYZ_yellow[0]< 8) and XYZ_yellow[2]<0.5 and XYZ_yellow[2]>-2:
                curr_red_id = len(self.fruit_database.red_fruit_arr_.markers)+100
                curr_yellow_id = len(self.fruit_database.yellow_fruit_arr_.markers)
                print(f"we have============================= {curr_red_id} ============================red fruits now")
                print(f"we have============================= {curr_yellow_id} ============================yellow fruits now")
                # self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))
                yellow_id = self.fruit_database.add_yellow_fruit_marker(color, curr_yellow_id, XYZ_yellow, abs(8), fruit_point.z)
                image_fruit_detections = self.draw_yellow_bbx(image_fruit_detections, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z), yellow_id)
                
                fruit_detections_image_msg = self.bridge.cv2_to_imgmsg(image_fruit_detections, "bgr8")
                fruit_detections_image_msg.header = odom_msg.header
                self.fruit_detections_pub.publish(fruit_detections_image_msg)

                self.fruit_database.publish_markers()
                cv2.putText(image_with_lidar_points, 'ADD ONE FRUIT', position, font, font_scale, color_g, thickness=2)


    def add_red_fruits_from_yolo_detection(self, image_mean_depth, means, yolo_fruits, color, uvd_points, image_with_lidar_points, odom_msg, image_fruit_detections):
        for fruit_point in yolo_fruits:
            # print(f"rpy_deg is : {rpy_deg}")
            print(f"red fuit is at ({int(fruit_point.x)},{int(fruit_point.y)}, {int(fruit_point.z)})")
            if(fruit_point.x<100 or fruit_point.x >540):
                return
            # fruit_point = PointStamped()
            # fruit_point = two_d_fruit_keypoints_msg.point
            image_with_lidar_points = self.draw_bbx(image_with_lidar_points, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z))

            # fruit_point.z = fruit_point.z
            # fruit_depth = self.find_yolo_fruit_depth(float(fruit_point.z))
            fruit_depth1, knn_points = self.find_red_fruit_depth(uvd_points, fruit_point)
            fruit_depth2 = self.find_yolo_red_depth(float(fruit_point.z))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",fruit_depth1)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",fruit_depth2)
            for i in range(0,len(knn_points)):
                # Draw the line
                p = knn_points
                cv2.line(image_with_lidar_points, (int(p[i][0]), int(p[i][1])), (int(fruit_point.x), int(fruit_point.y)), (0,0,255), 2)
            position1 = (int(fruit_point.x+20), int(fruit_point.y)-20) 
            position2 = (int(fruit_point.x+20), int(fruit_point.y)+20) 
            dpeth_str1 = str(fruit_depth1)
            cv2.putText(image_with_lidar_points, dpeth_str1, position1, self.font, self.font_scale, self.color_r, thickness=2)
            dpeth_str2 = str(fruit_depth2)
            cv2.putText(image_with_lidar_points, dpeth_str2, position2, self.font, self.font_scale, self.color_y, thickness=2)

            if(fruit_depth1 == None)or(fruit_depth1 == False):
                fruit_depth = fruit_depth1
                print('we have return here add_red_fruits_from_yolo_detection')
                return
            else:
                # fruit_depth1 = fruit_depth1
                fruit_depth = fruit_depth1 * 0.2 + fruit_depth2 * 0.8
            # XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
            
            print(f"has_valid_depth is {fruit_depth}")
            if(not fruit_depth):
                print(f"no valid depth")
                return
            elif(fruit_depth):
                print(f"has valid depth")
                if(image_mean_depth < 6):
                    depth_diff0 = abs(fruit_depth - means[0])
                    depth_diff1 = abs(fruit_depth - means[1])
                    if(depth_diff0 < depth_diff1):
                        fruit_depth = fruit_depth*0.3 + means[0]*0.7
                    else:
                        fruit_depth = fruit_depth*0.3 + means[1]*0.7

                XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color_g = (0, 255, 0)  # BGR color for blue
                color_r = (0, 0, 0)  # BGR color for blue
                # Position for the text
                position = (int(fruit_point.x), int(fruit_point.y))  # You can adjust this according to your needs

                # if(fruit_depth > 3 and fruit_depth < 9 and XYZ_yellow[0] > 2 and XYZ_yellow[0]< 8) and XYZ_yellow[2]<0.5 and XYZ_yellow[2]>-2:
                curr_red_id = len(self.fruit_database.red_fruit_arr_.markers)+100
                curr_yellow_id = len(self.fruit_database.yellow_fruit_arr_.markers)
                print(f"we have============================= {curr_red_id} ============================red fruits now")
                print(f"we have============================= {curr_yellow_id} ============================yellow fruits now")
                # self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))
                red_id = self.fruit_database.add_red_fruit_marker(color, curr_red_id, XYZ_yellow, abs(8), fruit_point.z)
                image_fruit_detections = self.draw_red_bbx(image_fruit_detections, color, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z), curr_red_id)
                
                fruit_detections_image_msg = self.bridge.cv2_to_imgmsg(image_fruit_detections, "bgr8")
                fruit_detections_image_msg.header = odom_msg.header
                self.fruit_detections_pub.publish(fruit_detections_image_msg)

                self.fruit_database.publish_markers()
                cv2.putText(image_with_lidar_points, 'ADD ONE FRUIT', position, font, font_scale, color_g, thickness=2)
                cv2.putText(image_with_lidar_points, 'ADD ONE FRUIT', position, font, font_scale, color_g, thickness=2)


    def calculate_gaussian(self, uvd_points, odom_msg):
        # Create histogram
        data = uvd_points[:,2]
        data = np.clip(data, 1, 12)

        # Fit a Gaussian Mixture Model with two components
        gmm = GaussianMixture(n_components=2, random_state=0).fit(data.reshape(-1, 1))

        # Get the fitted parameters
        means = gmm.means_.flatten()
        return means

    def lidar_msg_to_xyz_array(self, lidar_msg):
        assert isinstance(lidar_msg, PointCloud2)
        
        # Create a generator for the points
        point = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Create a list of the points
        points_arr = list(point)
        points_arr = np.array(points_arr)
        points_arr = points_arr[points_arr[:, 0] >= 1]
        points_arr = points_arr[points_arr[:, 0] <= 7.5]
        points_arr = points_arr[points_arr[:, 1] <2.5]
        points_arr = points_arr[points_arr[:, 1] >-2.5]

        # Get the size of the list
        num_points = len(points_arr)
        print(f"we should return here, num_points is {num_points}")
        if num_points < 20:
            # Create an 8x3 array filled with ones
            new_points = np.ones((200, 3))
            return new_points
        
        # Convert the list to a numpy array and return it
        points_arr = np.array(points_arr)
        print(f"================points_arr.shape: {points_arr.shape}")
        # Check the shape of points_arr
        print(f"the size of points_arr before filter by xyz is: {points_arr.shape}")
        
        # points_arr = points_arr[points_arr[:, 0] >= 1]
        # points_arr = points_arr[points_arr[:, 1] <10]
        # points_arr = points_arr[-10< points_arr[:, 1]]

        return np.array(points_arr)     
    

    def lidar_points_to_uvd(self, points_arr):
        rotated_points_arr = np.dot(self.R_cam_lidar, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_lidar_g
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]

        return uvd_points
    
    
    
    # 根据fruit_size 调整需要的knn中k的数量

    def find_yolo_yellow_depth(self, fruit_size):
        # yolo_fruit_depth = 70 / fruit_size #tz param
        yolo_fruit_depth = 58 / fruit_size  # ywy param
        return yolo_fruit_depth

    def find_yolo_red_depth(self, fruit_size):
        # yolo_fruit_depth = 70 / fruit_size #tz param
        yolo_fruit_depth = 50 / fruit_size  # ywy param
        return yolo_fruit_depth
    
    # fruit_point 包括 (x,y,size)
    # 根据 size 判断是否合理
    def find_red_fruit_depth(self, uvd_points, fruit_point):
        # Calculate the Manhattan distance between each point in uvd_points and fruit_point
        knn_points = self.find_knn(uvd_points, fruit_point)
        
        fruit_size = fruit_point.z
        if(fruit_size < self.min_red_size):
            self.min_red_size = fruit_size
        if(fruit_size > self.max_red_size):
            self.max_red_size = fruit_size
        print(f"the min red size is: {self.min_red_size}")
        print(f"the max red size is: {self.max_red_size}")


        knn_points_noOutlier = self.remove_outliers(knn_points)
        weighted_depth, depth_mean, depth_var = self.find_depth_mean_var(knn_points_noOutlier, fruit_point)
        print(f"mean of depth is: {depth_mean};\nVariance of depth is: {depth_var}")
        print(f"mean of weighted_depth is: {weighted_depth};\nVariance of depth is: {depth_var}")
        if((depth_var/fruit_size) > 0.66):
            print(f"the dpeth_var is {depth_var}/{fruit_size} = {depth_var/fruit_size}, TOO LARGE, return")
            return False, knn_points_noOutlier
        else:
            fruit_depth = depth_mean
            fruit_depth = weighted_depth
            print(f"valid_var, the fruit_depth is set as {weighted_depth}")
        
        has_valid_depth = self.check_knn_dist(knn_points_noOutlier, fruit_point)
        if(not has_valid_depth):
            fruit_depth = False
            return False, knn_points_noOutlier
        else:
            pass
        if((fruit_depth*fruit_size)>72)or((fruit_depth*fruit_size)<28):
            print(f"fruit_depth*fruit_size>75, return false")
            fruit_depth = False
       
        # d3size = fruit_size*(fruit_depth**3)
        # if(d3size > 2500):
        #     print(f"fruit_size*(fruit_depth**3) is {d3size} > 2500, return false")
        #     fruit_depth = False

        return fruit_depth, knn_points_noOutlier
    
    def find_yellow_fruit_depth(self, uvd_points, fruit_point):
        # Calculate the Manhattan distance between each point in uvd_points and fruit_point
        print('inside find_yellow_fruit_depth()')
        knn_points = self.find_knn(uvd_points, fruit_point)
        fruit_size = fruit_point.z
        if(fruit_point.z < self.min_yellow_size):
            self.min_yellow_size = fruit_point.z
        if(fruit_point.z > self.max_yellow_size):
            self.max_yellow_size = fruit_point.z
        print(f"the min yellow size is: {self.min_yellow_size}")
        print(f"the max yellow size is: {self.max_yellow_size}")

        knn_points_noOutlier = self.remove_outliers(knn_points)
        weighted_depth, depth_mean, depth_var = self.find_depth_mean_var(knn_points_noOutlier, fruit_point)
        print(f"mean of depth is: {depth_mean};\nVariance of depth is: {depth_var}")
        print(f"mean of weighted_depth is: {weighted_depth};\nVariance of depth is: {depth_var}")
        if((depth_var/fruit_size) > 0.66):
            print(f"the dpeth_var is {depth_var}/{fruit_size} = {depth_var/fruit_size}, TOO LARGE, return")
            return False, knn_points_noOutlier
        else:
            fruit_depth = depth_mean
            fruit_depth = weighted_depth
            print(f"valid_var, the fruit_depth is set as {weighted_depth}")
        
        has_valid_depth = self.check_knn_dist(knn_points_noOutlier, fruit_point)
        if(not has_valid_depth):
            fruit_depth = False
            return False, knn_points_noOutlier
        else:
            pass
        # if((fruit_depth*fruit_size)>70)or((fruit_depth*fruit_size)<35):# bag4 70 35
        #     print(f"fruit_depth*fruit_size>70, return false")
        #     fruit_depth = False

        return fruit_depth, knn_points_noOutlier
    
    def check_knn_dist(self, knn_points, fruit_point):
        fruit_size = int(fruit_point.z)
        distances = np.sqrt((knn_points[:, 0] - fruit_point.x)**2 + (knn_points[:, 1] - fruit_point.y)**2)
        if(distances[0] > (fruit_size*0.6)):
            print(f"\ndistance[0] is {distances[0]}, larger than fruit_size*0.6->{(fruit_size*0.6)}, False")
            return False
        else:
            return True


    def find_knn(self, uvd_points, fruit_point):
        print('inside find_knn')
        # Calculate the Euclidean distance between each point in uvd_points and fruit_point
        distances = np.sqrt((uvd_points[:, 0] - fruit_point.x)**2 + (uvd_points[:, 1] - fruit_point.y)**2)
        fruit_size = int(fruit_point.z)
        print(f"\nKNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN")
        print(f"fruit_size inside find_knn(fruit_point.z) is {fruit_size}")
        # Get the indices of the points sorted by their distances to fruit_point
        sorted_indices = np.argsort(distances)
        nearest_dist = distances[sorted_indices[0]]
        # nearest_dist_v = abs(uvd_points[sorted_indices[0],1]-fruit_point.y)
        # print(f"nearest_dist is {nearest_dist_v}")
        # if nearest_dist > 1:
        if nearest_dist > fruit_size*0.2:
            print("we have returned empty knn_points here")
            has_valid_depth = False
        else:
            has_valid_depth = True
            print(f"set has_valid_depth as {has_valid_depth}")
        # Select the top k points
        # 不同水果大小，需要的 knn 中 k 的数量根据 fruit_size 改变
        knn_indices = sorted_indices[:(int(fruit_size)+4)]
        knn_points = uvd_points[knn_indices]
        knn_points_dist = distances[knn_indices]
        # knn_points_in_fruit_size = []
        knn_points = np.round(knn_points, 2)
        print(f"the fruit_point is at ({fruit_point.x:.2f},{fruit_point.y:.2f}), fruit_size is {fruit_point.z:.2f}")
        for i in range(0,len(knn_points)-1):
            pxiel_dist = abs(uvd_points)
            # print(f"{i}th point is {knn_points[i]}, and pixel dist is {knn_points_dist[i]:.2f}")
        return knn_points
    
    def remove_outliers(self, knn_points):
        # Extract the third element from each point
        print(f"inside remove_outliers")
        third_elements = [point[2] for point in knn_points]

        # Calculate the IQR of the third elements
        q1 = np.percentile(third_elements, 25)
        q3 = np.percentile(third_elements, 75)
        iqr = q3 - q1
        # Identify the outlier indices
        outlier_indices = [i for i, x in enumerate(third_elements) if x < (q1 - 1.5 * iqr) or x > (q3 + 1.5 * iqr)]
        # Remove the outliers from the points
        knn_points_without_outliers = [point for i, point in enumerate(knn_points) if i not in outlier_indices]
        knn_points_without_outliers = np.array(knn_points_without_outliers)
        print(f"knn_points_without_outliers :{knn_points_without_outliers}")
        return knn_points_without_outliers

    def weighted_mean(self, knn_points, pixel_distances):
        weights = [1 / dist for dist in pixel_distances]
        weighted_sum = sum(p * w for p, w in zip(knn_points, weights))
        total_weight = sum(weights)
        weighted_depth = weighted_sum[2] / total_weight
        return weighted_depth

    def find_depth_mean_var(self, knn_points, fruit_point):
        # Extract the depth values
        depths = knn_points[:, 2]
        print(f"depths are {depths}")
        # Calculate the variance
        variance_depth = np.var(depths)
        mean_dpeth = np.mean(depths)
        distances = np.sqrt((knn_points[:, 0] - fruit_point.x)**2 + (knn_points[:, 1] - fruit_point.y)**2)
        weighted_depth = self.weighted_mean(knn_points, distances)

        return weighted_depth, mean_dpeth, variance_depth

    def find_mode_depth(self, knn_points, fruit_size):
        # Extract the depth values
        depths = knn_points[:, 2]
        # Round the depths to one decimal place
        rounded_depths = np.round(depths, 0)
        print(f"rounded depth is: {rounded_depths}")
        # Calculate the mode
        mode_depth = statistics.mode(rounded_depths)
        # Calculate the mode
        mode_depth = statistics.mode(rounded_depths)
        # Count the occurrences of the mode
        mode_count = np.count_nonzero(rounded_depths == mode_depth)
        # Calculate 80% of the number of points
        eighty_percent_count = (self.knn_mode_percent/100) * len(depths)
        # Check if the mode count is greater than or equal to 80% of the number of points
        if mode_count >= eighty_percent_count:
            print(f"mode depth larger than {self.knn_mode_percent}%, mode is {mode_depth}")
            return mode_depth
        else:
            print(f"mode depth less than {self.knn_mode_percent}%, discard")
            return False

    def colorize_uvd_visualization(self, uvd_points, image):
        # Convert xy from homogeneous points to image points, keep z as its real dist
        mask = (uvd_points[:, 0] >= 40) & (uvd_points[:, 0] <= 600)  #changed but not tested

        # Apply the mask to the uvd_points array to get the filtered array
        filtered_uvd_points = uvd_points[mask]
        filtered_z_values = filtered_uvd_points[:, 2]
        image_mean_depth = np.mean(filtered_z_values)
        # # Get the first column of uvd_points
        # first_column = uvd_points[:, 0]

        middle_u_mask = (uvd_points[:, 0] >= 300) & (uvd_points[:, 0] <= 380)  #changed but not tested
        middle_filtered_uvd_points = uvd_points[middle_u_mask]
        middle_filtered_z_values = middle_filtered_uvd_points[:, 2]
        middle_image_mean_depth = np.mean(middle_filtered_z_values)
        # print(f"the fov_mean_depth is:{image_mean_depth}")
        # Clip the z_values to the range [3, 8]
        z_values = uvd_points[:,2]
        z_values = np.clip(z_values, 3, 9)
        # Normalize the z_values to the range [0, 1]
        z_values = (z_values - z_values.min()) / (z_values.max() - z_values.min())
        # Define the colors for close and far points
        close_color = np.array([0, 0, 255])  # Red in BGR
        far_color = np.array([255, 125, 0])  # Blue in BGR

        # Calculate the colors for the points
        colors = (1 - z_values[:, np.newaxis]) * close_color + z_values[:, np.newaxis] * far_color
        # Concatenate the image points and colors
        colored_image_points = np.hstack((uvd_points, colors))

        for x, y, z, b,g,r in np.int32(colored_image_points):
            # print(f"b,g,r:{b},{g},{r}")
            cv2.circle(image, (x, y), 1, color=(int(b), int(g), int(r)), thickness=2)
        
        # 画出分割线
        width_step = 100
        height_step = 100
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        path_length_str = "path_length: " + str(self.path_length)
        cv2.putText(image, path_length_str,(300, 50), font, font_scale, (0,0,0), thickness)
        str_middle_image_mean_depth = "300-380 depth: " + str(middle_image_mean_depth)
        cv2.putText(image, str_middle_image_mean_depth,(300, 80), font, font_scale, (0,0,0), thickness)
        str_roll = "roll: " + str(self.roll)
        cv2.putText(image, str_roll,(300, 120), font, font_scale, (0,0,0), thickness)
        str_roll = "picth: " + str(self.pitch)
        cv2.putText(image, str_roll,(300, 150), font, font_scale, (0,0,0), thickness)

        # 中轴十字线
        cv2.line(image, (320, 0), (320, image.shape[0]), (255, 255, 50), 2)
        cv2.line(image, (0, 240), (640, 240), (255, 255, 50), 2)
        for i in range(1, 7):
            cv2.line(image, (i * width_step, 0), (i * width_step, image.shape[0]), (255, 255, 255), 1)
            cv2.line(image, (0, i * height_step), (image.shape[1], i * height_step), (255, 255, 255), 1)

        return image, image_mean_depth, middle_image_mean_depth
    
    def fruit_markers_to_uvd(self, odom_msg):
        print("fruit_markers_to_uvd() called")
        # Loop through each marker in the MarkerArray
        # Initialize an empty list to hold the marker positions
        worldXYZ_fruit = np.empty((0,3))
        print("=================yellow_fruits===========================")
        print(f"the len(self.yellow_fruit_arr_.markers is {len(self.fruit_database.yellow_fruit_arr_.markers)})")
        print(f"=================yellow_prob_mean is {self.fruit_database.yellow_prob_mean}===========================")
        for i in range(0,len(self.fruit_database.yellow_fruit_arr_.markers)):
            marker = self.fruit_database.yellow_fruit_arr_.markers[i]
            marker_position = marker.pose.position
            new_fruit = np.array([float(marker_position.x), marker_position.y, marker_position.z])
            print(f"{i}th marker position is ({new_fruit}), the prob is {marker.color.a}")
            # Append the marker's position to the list
            worldXYZ_fruit = np.append(worldXYZ_fruit, [new_fruit], axis=0)
        
        print("===================red_fruits===========================")
        print(f"the len(self.red_fruit_arr_.markers is {len(self.fruit_database.red_fruit_arr_.markers)})")
        print(f"====================red_prob_mean is {self.fruit_database.red_prob_mean}===========================")
        for i in range(0,len(self.fruit_database.red_fruit_arr_.markers)):
            marker = self.fruit_database.red_fruit_arr_.markers[i]
            marker_position = marker.pose.position
            new_fruit = np.array([float(marker_position.x), marker_position.y, marker_position.z])
            print(f"{i}th marker poistion is ({new_fruit}), the prob is {marker.color.a}")
            # Append the marker's position to the list
            worldXYZ_fruit = np.append(worldXYZ_fruit, [new_fruit], axis=0)
        # print(f"worldXYZ_fruit is:\n{worldXYZ_fruit}")
        imuXYZ_fruits = self.transform_utils.Timu_world(worldXYZ_fruit, odom_msg)
        rotated_lidarXYZ_fruits = np.dot(self.R_lidar_imu, imuXYZ_fruits.T)
        translated_rotated_lidarXYZ_fruits = rotated_lidarXYZ_fruits.T + self.transvec_lidar_imu_g
        # print(f"translated_rotated_lidarXYZ_fruits is:\n{translated_rotated_lidarXYZ_fruits}")
        # homo_fruits = np.dot(self.camera_matrix, translated_rotated_lidarXYZ_fruits.T).T
        # uvd_fruits = homo_fruits[:, :3] / homo_fruits[:, 2, np.newaxis]
        # uvd_fruits[:,2] = homo_fruits[:,2]
        uvd_fruits = self.lidar_fruits_to_uvd(translated_rotated_lidarXYZ_fruits)
        return uvd_fruits
    
    def lidar_fruits_to_uvd(self, points_arr):
        rotated_points_arr = np.dot(self.R_cam_lidar, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_imu_g
        # print(f"translated_rotated_points_arr is {translated_rotated_points_arr}")
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]

        return uvd_points

    def colorize_fruits_uvd(self, uvd_fruits, image):
        # Convert xy from homogeneous points to image points, keep z as its real dist
        z_values = uvd_fruits[:, 2]
        # print(f"the fov_mean_depth is:{image_mean_depth}")
        # Clip the z_values to the range [3, 8]
        # Define the colors for close and far points
        close_color = np.array([0, 0, 0])  # Red in BGR
        far_color = np.array([0, 0, 0])  # Blue in BGR
        # Calculate the colors for the points
        colors = (1 - z_values[:, np.newaxis]) * close_color + z_values[:, np.newaxis] * far_color
        # Concatenate the image points and colors
        colored_image_points = np.hstack((uvd_fruits, colors))
        for x, y, z, b,g,r in np.int32(colored_image_points):
            # print(f"b,g,r:{b},{g},{r}")
            # print(f"the fruit ponit is at({x},{y})")
            cv2.circle(image, (x, y), 15, color=(int(b), int(g), int(r)), thickness=11)
        
        return image
    
    def draw_bbx(self, image, color_string, x, y, size):
        # draw the fruit first
        color = (0, 255, 0)

        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        # Add a text label with a background color
        label = color_string
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-15, y1
        if(color_string == 'yellow'):
            image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,255,255), thickness=3)
            cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,255,255), -1)
            cv2.putText(image, "Yellow", (text_x, text_y), font, font_scale, (0,0,0), thickness)
        if(color_string == 'red'):
            image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,0,255), thickness=3)
            red_text_x =text_x+3
            cv2.rectangle(image, (red_text_x, text_y - text_size[1]), (red_text_x + text_size[0], text_y), (0,0,255), -1)
            cv2.putText(image, "Red", (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    def draw_red_bbx(self, image, color_string, x, y, size, red_id):
        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        # Add a text label with a background color
        label = "Red"+str(red_id - 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-10, y1

        # image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,0,255), thickness=3)
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,0,255), -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    def draw_yellow_bbx(self, image, color_string, x, y, size, yellow_id):
        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w = size
        h = size
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 255), 2)
        # Add a text label with a background color
        label = "Yellow"+str(yellow_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1-20, y1
        # image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,255,255), thickness=3)
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0,255,255), -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
    def publish_camera_fov_marker(self, odom_msg):
        marker = Marker()
        marker.header.frame_id = "camera_init"  # Change to your camera frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_fov"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker pose (same as camera)
        marker.pose = odom_msg.pose.pose

        # Set scale and color
        marker.scale.x = 0.1  # Line width
        marker.color.r = 0.20
        marker.color.b = 1.0
        marker.color.g = 0.99
        marker.color.a = 1.0

        # Define FoV dimensions
        fov_height = 3.0  # Change to your FoV height
        fov_width = 4.0  # Change to your FoV width

        # Define points of the pyramid (assuming camera at origin)
        p1 = Point(0, 0, 0)  # Camera position
        p10 = Point(5, 0, 0)
        p2 = Point(5, fov_width / 2, fov_height / 2)
        p3 = Point(5, -fov_width / 2, fov_height / 2)
        p4 = Point(5, -fov_width / 2, -fov_height / 2)
        p5 = Point(5, fov_width / 2, -fov_height / 2)



        p6 = Point(5, 0, -fov_height / 2)
        p7 = Point(5, 0, fov_height / 2)
        p8 = Point(5, fov_width / 2, 0)
        p9 = Point(5, -fov_width / 2, 0)

        p11 = Point(3, fov_width / 2, fov_height / 2)
        p12 = Point(3, -fov_width / 2, fov_height / 2)
        p13 = Point(3, -fov_width / 2, -fov_height / 2)
        p14 = Point(3, fov_width / 2, -fov_height / 2)

        p15 = Point(7, fov_width / 2, fov_height / 2)
        p16 = Point(7, -fov_width / 2, fov_height / 2)
        p17 = Point(7, -fov_width / 2, -fov_height / 2)
        p18 = Point(7, fov_width / 2, -fov_height / 2)
        # Add lines to the marker
        marker.points = [p1, p10, p1, p2, p1, p3, p1, p4, p1, p5, p2, p3, p3, p4, p4, p5, p5, p2, p6, p7, p8, p9, p11, p12, p12, p13, p13, p14, p14, p11, p15,p16,p16,p17,p17,p18,p18,p15]

        # Publish the marker
        self.cam_fov_pub.publish(marker)

        # rospy.init_node('camera_fov_visualizer')
        # marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    
    def publish_quad_rviz(self, odom_msg):
        # 创建 MarkerArray
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "camera_init"  # Change to your camera frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "quad"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Set marker pose (same as camera)
        marker.pose = odom_msg.pose.pose

        # Set scale and color
        marker.scale.x = 0.1  # Line width
        marker.color.r = 0.20
        marker.color.b = 1.0
        marker.color.g = 0.199
        marker.color.a = 1.0

        # Define FoV dimensions

        # Define points of the pyramid (assuming camera at origin)
        p1 = Point(0, 0, 0)  # Camera position
        p2 = Point(1, 1, 0)
        p3 = Point(1, -1, 0)
        p4 = Point(-1, 1, 0)
        p5 = Point(-1, -1, 0)
        # Add lines to the marker
        marker.points = [p1, p2, p1, p3, p1, p4, p1, p5]
        marker_array.markers.append(marker)

        # 创建四个圆盘
        quad_center = odom_msg.pose.pose.position
        quad_orientation = odom_msg.pose.pose.orientation
        for i, pos in enumerate([(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]):
            marker_disc = Marker()
            marker_disc.header.frame_id = "camera_init"
            marker_disc.ns = "quadrotor"
            marker_disc.id = i + 1
            marker_disc.type = Marker.CYLINDER
            marker_disc.action = Marker.ADD
            marker_disc.pose.position = Point(quad_center.x+pos[0], quad_center.y+pos[1], quad_center.z+pos[2])
            marker_disc.pose.orientation = Quaternion(quad_orientation.x, quad_orientation.y, quad_orientation.z, quad_orientation.w)
            marker_disc.scale.x = 0.9  # 直径
            marker_disc.scale.y = 0.9  # 直径
            marker_disc.scale.z = 0.31  # 高度
            marker_disc.color.a = 1.0  # alpha
            marker_disc.color.r = 1.0  # red
            marker_array.markers.append(marker_disc)


        # 发布 MarkerArray
        self.quad_rviz_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        lr = LidarReprojector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
