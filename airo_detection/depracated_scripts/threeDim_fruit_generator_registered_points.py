import rospy
import cv2
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
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
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_matrix





import sys, os
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)
from threeDim_fruit_database import *
from twoDim_fruit_detector_ywy import TwoDFruitDetector
# from twoD_fruit_detector import TwoDFruitDetector
from transform_utils import *
import tf

# @TODO 用registered points 点云太稀疏，很多水果都看不到

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
        self.knn_mode_percent = 70
        # self.knn_dist = 2
        self.bridge = CvBridge()
        self.fx = 624.325
        self.fy = 624.4396
        self.cx = 320.5857
        self.cy = 216.9464
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
        self.R_cam_IMU = np.dot(self.R_cam_lidar, self.R_lidar_imu)
        # self.R_cam_lidar = np.dot(np.array([[0,1,0], [-1, 0, 0], [0,0.,1]]), np.array([[-0.2419219,  0.0000000, -0.9702957], [0.0000000,  1.0000000,  0.0000000], [0.9702957,  0.0000000, -0.2419219]]))
        self.rotvec_cam_lidar_g, _ = cv2.Rodrigues(self.R_cam_lidar)
        self.transvec_cam_lidar_g = np.array([0.1,0,-0.1])
        self.lidar_projected_image_pub = rospy.Publisher('/fused_image_ooad', Image, queue_size=10)

        self.twoD_fruit_detector = TwoDFruitDetector()
        self.fruit_database = PlantFruitDatabase()
        self.transform_utils = TransformUtils()

        # ROS node and subscriber
        rospy.init_node('lidar_reprojector', anonymous=True)
        # rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)

        # Subscribe to both topics
        self.sub_lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        self.sub_undistorted_lidar = message_filters.Subscriber('/cloud_registered', PointCloud2)
        self.sub_image = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        # self.sub_2dfruit = message_filters.Subscriber('two_d_fruit_keypoints', PointStamped)
        # self.sub_odom = message_filters.Subscriber('/kiss/odometry', Odometry)
        self.sub_odom = message_filters.Subscriber('/Odometry', Odometry)

        # Synchronize the topics with a slop of 0.1 seconds
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_lidar, self.sub_odom], 30, 0.075)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_image, self.sub_undistorted_lidar, self.sub_odom], 30, 0.075)
        self.ts.registerCallback(self.callback)

         # Create queues for lidar_msg and odom_msg
        self.lidar_msg_queue = queue.Queue(maxsize=self.queue_size)
        self.odom_msg_queue = queue.Queue(maxsize=self.queue_size)


    def callback(self, image_msg, lidar_msg, odom_msg):
        # Convert ROS PointCloud2 to PCL
         # Check if queues are full
        # if(lidar_msg.header.stamp < image_msg.header.stamp):
        #     print(f"lidar time: {lidar_msg.header.stamp}")
        #     print(f"image time: {image_msg.header.stamp}")
        #     print(f"lidar time earlier than image time, return")
        #     return
        # else:
        #     print(f"lidar time: {lidar_msg.header.stamp}")
        #     print(f"image time: {image_msg.header.stamp}")
        #     print(f"lidar time smaller than image time, pass=======")

        self.lidar_msg_queue.put(lidar_msg)
        if self.lidar_msg_queue.full():
            # Remove the oldest message (first in)
            self.lidar_msg_queue.get()

        self.odom_msg_queue.put(odom_msg)
        if self.odom_msg_queue.full():
            # Remove the oldest message (first in)
            self.odom_msg_queue.get()

        # print(f"sefl.lidar_msg_queue.qsize() {self.lidar_msg_queue.qsize()}")
        # print("we are inside the callback now")

        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        rpy_deg = odom_msg_to_rpy(odom_msg)
        # print(f"rpy is {rpy_deg}\n")
        # if(abs(rpy_deg[1])<2.5 and abs(rpy_deg[0])<30):
        if(abs(rpy_deg[1])<3.5):
            # print(f"GOOD ROLL, PASS-> roll is {rpy_deg[0]}")
            # print(f"GOOD PITCH, PASS->picth is {rpy_deg[1]}\n")
            pass
        else:
            # print(f"BAD ROLL, RETURN-> roll is {rpy_deg[0]}")
            # print(f"BAD PITCH, RETURN->picth is {rpy_deg[1]}\n\n\n")
            return

        # Reproject points
        # points_arr = self.lidar_msg_to_xyz_array(lidar_msg)
        points_arr = self.lidar_msg_queue_to_xyz_array()

        # @TODO points from world coor to IMU coor
        points_body = self.T_body_world_points(points_arr, odom_msg)

        uvd_points = self.lidar_points_to_uvd(points_body)  # YWY noly need uvd
        # self.plot_depth_histagram_with_gaussian(uvd_points, odom_msg)  #
        # self.twoD_fruit_detector.detect_fruit(image)
        # image_with_lidar_points = self.colorize_uvd_visualization(uvd_points, image)  #colored_image_points only for visualization, could delete
        # for fruit_point in self.twoD_fruit_detector.fruit_points_:
        fruit_points = self.twoD_fruit_detector.detect_fruit(image)
        image_with_lidar_points = self.colorize_uvd_visualization(uvd_points, image)  #colored_image_points only for visualization, could delete
        for fruit_point in fruit_points:
        # fruit_point = PointStamped()
            # fruit_point = two_d_fruit_keypoints_msg.point
            image = self.draw_bbx(image, int(fruit_point.x),int(fruit_point.y), int(fruit_point.z))

            # fruit_depth = self.find_fruit_depth(uvd_points, fruit_point)
            # XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
            
            fruit_depth = self.find_fruit_depth(uvd_points, fruit_point)
            # has_valid_depth = False
            # knn_points = self.find_knn(uvd_points, fruit_point, 3, int(fruit_point.z/3))
            print(f"has_valid_depth is {fruit_depth}")
            if(not fruit_depth):
                print(f"no valid depth")
            elif(fruit_depth):
                print(f"has valid depth")
                # mode_depth = self.find_mode_depth(knn_points, 80)
                # fruit_depth = mode_depth
                # print(f"MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH={mode_depth}")
                XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color_g = (0, 255, 0)  # BGR color for blue
                color_r = (0, 0, 0)  # BGR color for blue
                # Position for the text
                position = (int(fruit_point.x), int(fruit_point.y))  # You can adjust this according to your needs

                # if(fruit_depth > 3 and fruit_depth < 9 and XYZ_yellow[0] > 2 and XYZ_yellow[0]< 8) and XYZ_yellow[2]<0.5 and XYZ_yellow[2]>-2:
                if(fruit_depth > 1 and fruit_depth < 10):
                    curr_yellow_id = len(self.fruit_database.fruit_arr_.markers)
                    print(f"we have============================= {curr_yellow_id} ============================yellow fruits now")
                    # self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))
                    self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(8), fruit_point.z/2)
                    self.fruit_database.publish_markers()
                    cv2.putText(image_with_lidar_points, 'ADD ONE FRUIT', position, font, font_scale, color_g, thickness=2)
                    # print("BINGO\n")
                    # print(f"fruit depth is:{fruit_depth}")
                else:
                    # print(f"the fruit_depth is {fruit_depth}, discarded\n CHIOBO\n\n")
                    # Set the font, scale, and color
                    # Add the text to the image
                    cv2.putText(image_with_lidar_points, 'this fruit \nis invalid', position, font, font_scale, color_r, thickness=4)
                    colored_image_msg = self.bridge.cv2_to_imgmsg(image_with_lidar_points, "bgr8")
                    colored_image_msg.header = odom_msg.header
                    self.lidar_projected_image_pub.publish(colored_image_msg)

        colored_image_msg = self.bridge.cv2_to_imgmsg(image_with_lidar_points, "bgr8")
        colored_image_msg.header = odom_msg.header
        self.lidar_projected_image_pub.publish(colored_image_msg)


        # image.clear()
        # uvd_points.clear()


    def odom_to_transform_matrix(self, odom_msg):
        # Get the position and orientation from the Odometry message
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        # Convert the orientation from a quaternion to a rotation matrix
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation_matrix = quaternion_matrix(quaternion)[:3, :3]

        # Create the translation vector
        translation_vector = np.array([position.x, position.y, position.z])

        # Create the transformation matrix
        T_body_world = np.eye(4)
        T_body_world[:3, :3] = rotation_matrix
        T_body_world[:3, 3] = translation_vector

        return T_body_world

    def T_body_world_points(self, points_world, odom_msg):
        # Convert the Odometry message to a transformation matrix
        T_body_world = self.odom_to_transform_matrix(odom_msg)

        # Transform the points to the body frame
        points_body = []
        for point_world in points_world:
            # Convert the point to homogeneous coordinates
            point_world_hom = np.append(point_world, 1)

            # Apply the transformation
            point_body_hom = np.dot(T_body_world, point_world_hom)

            # Convert back to non-homogeneous coordinates
            point_body = point_body_hom[:3]
            points_body.append(point_body)
        
        points_body = np.array(points_body)
        print(f"cloud_registered.shape is:{points_body.shape}" )
        return np.array(points_body)


    def plot_depth_histagram_with_gaussian(self, uvd_points, odom_msg):
        # Create histogram
        data = uvd_points[:,2]
        data = np.clip(data, 1.2, 12)

        # Fit a Gaussian Mixture Model with two components
        gmm = GaussianMixture(n_components=3, random_state=0).fit(data.reshape(-1, 1))

        # Get the fitted parameters
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        # Create an array of x values for plotting
        x = np.linspace(data.min(), data.max(), 1000)

        # Create a figure
        fig, ax = plt.subplots()

        # Plot the histogram
        ax.hist(data, bins=10, density=True, edgecolor='black', alpha=0.5)

        # Plot each Gaussian
        for mean, cov, weight in zip(means, covariances, weights):
            # ax.plot(x, weight*norm.pdf(x, mean, np.sqrt(cov)))
            ax.plot(x, norm.pdf(x, mean, np.sqrt(cov)))

        # Save plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)


        # Convert PNG in memory to OpenCV mat
        img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
        buf.close()

        rpy_deg = odom_msg_to_rpy(odom_msg)
        print(f"rpy is {rpy_deg}\n")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_r = (0, 0, 255)  # BGR color for blue
        # Position for the text
        position = (int(400), int(50))  # You can adjust this according to your needs
        cv2.putText(img, f'Pitch:{rpy_deg[1]}', position, font, font_scale, color_r, thickness=2)

        # Convert mat to ROS image
        bridge = CvBridge()
        image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")

        # Publish the image message
        histagram_image_pub = rospy.Publisher("/depth_histagram", Image, queue_size=10)
        histagram_image_pub.publish(image_message)

        # Overwrite img with a black image
        img[:] = 0

    def lidar_msg_to_xyz_array(self, lidar_msg):
        assert isinstance(lidar_msg, PointCloud2)
        
        # Create a generator for the points
        point = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Create a list of the points
        points_arr = list(point)
        
        # Convert the list to a numpy array and return it
        points_arr = np.array(points_arr)
        print(f"================points_arr.shape: {points_arr.shape}")
        points_arr = points_arr[points_arr[:, 0] >= 1]
        points_arr = points_arr[points_arr[:, 1] <10]
        points_arr = points_arr[-10< points_arr[:, 1]]

        return np.array(points_arr)
    
    def lidar_msg_queue_to_xyz_array(self):
        # if(self.lidar_msg_queue.empty()):
            # return
        # Initialize an empty array to store all points
        all_points_arr = np.array([])

        # Create a list to store lidar messages and odometry messages
        # lidar_msgs = []
        # odom_msgs = []

        # While there are messages in the queues
        # while not self.lidar_msg_queue.empty() and not self.odom_msg_queue.empty():
        #     # Get the lidar_msg and odom_msg from the queues
        #     lidar_msg = self.lidar_msg_queue.get()
        #     odom_msg = self.odom_msg_queue.get()

        #     # Store the messages in the lists
        #     lidar_msgs.append(lidar_msg)
        #     odom_msgs.append(odom_msg)

        # Get the transformation from the 10th message's coordinate system to the 5th message's coordinate system
        # Assuming you have a function get_relative_transformation that takes two odometry messages and returns the relative transformation

        # Apply the relative transformation to all points in the 5th lidar scan
        # Convert the list to a numpy array and return it
        point = point_cloud2.read_points(self.lidar_msg_queue.queue[self.lidar_msg_queue.qsize()-1], field_names=("x", "y", "z"), skip_nans=True)
        # Create a list of the points
        points_arr = list(point)
        points_arr = np.array(points_arr)
        points_arr = points_arr[points_arr[:, 0] >= 1]
        points_arr = points_arr[points_arr[:, 1] <10]
        points_arr = points_arr[-10< points_arr[:, 1]]
        all_points_arr = np.array(points_arr)
        # print(f"================all_points_arr.shape: {all_points_arr.shape}")


        for i in range(0, self.lidar_msg_queue.qsize()-1):
            # assert isinstance(lidar_msgs[i], PointCloud2)
            # assert isinstance(odom_msgs[i], Odometry)
            rot_mat_curr_pre, trans_vec_curr_pre = self.get_relative_transformation(self.odom_msg_queue.queue[i], self.odom_msg_queue.queue[self.odom_msg_queue.qsize()-1])

            # Create a generator for the points
            point = point_cloud2.read_points(self.lidar_msg_queue.queue[i], field_names=("x", "y", "z"), skip_nans=True)

            # Create a list of the points
            points_arr = list(point)

            # Convert the list to a numpy array
            points_arr = np.array(points_arr)
            # print(f"================points_arr.shape: {points_arr.shape}")


            # If this is the 5th message, apply the relative transformation
            # if i == 4:
            # rotated_points_arr-> nx3
            rotated_points_arr = np.dot(rot_mat_curr_pre, points_arr.T).T
            # print(f"================rotated_points_arr.shape: {rotated_points_arr.shape}")

            trans_vec_curr_pre = np.reshape(trans_vec_curr_pre, (3,1))
            # translated_rotated_points_arr = rotated_points_arr.T + trans_vec_curr_pre
            # rotated_points_arr-> nx3
            translated_rotated_points_arr = (rotated_points_arr.T + trans_vec_curr_pre).T
            # print(f"trans_vec_curr_pre.shape {trans_vec_curr_pre.shape}")
            # Append the transformed points to all_points_arr
            if all_points_arr.size == 0:
                all_points_arr = np.reshape(points_arr, (3,1))
            else:
                all_points_arr = np.concatenate((all_points_arr, translated_rotated_points_arr), axis=0)

        # Filter the points
        

        all_points_arr = np.array(all_points_arr)
        all_points_arr = all_points_arr[all_points_arr[:, 0] >= 1]
        all_points_arr = all_points_arr[all_points_arr[:, 1] <10]
        all_points_arr = all_points_arr[-10< all_points_arr[:, 1]]
        queue_points_arr = all_points_arr

        return np.array(queue_points_arr)
    
    def get_relative_transformation(self, curr_odom_msg, pre_odom_msg):
        # Extract the rotation and translation from the odometry messages
        # Assuming the rotation is represented as a quaternion and the translation as a vector in the odometry message
        pre_rot_quat = [pre_odom_msg.pose.pose.orientation.x, pre_odom_msg.pose.pose.orientation.y, pre_odom_msg.pose.pose.orientation.z, pre_odom_msg.pose.pose.orientation.w]
        pre_trans_vec = [pre_odom_msg.pose.pose.position.x, pre_odom_msg.pose.pose.position.y, pre_odom_msg.pose.pose.position.z]

        curr_rot_quat = [curr_odom_msg.pose.pose.orientation.x, curr_odom_msg.pose.pose.orientation.y, curr_odom_msg.pose.pose.orientation.z, curr_odom_msg.pose.pose.orientation.w]
        curr_trans_vec = [curr_odom_msg.pose.pose.position.x, curr_odom_msg.pose.pose.position.y, curr_odom_msg.pose.pose.position.z]

        # Convert the quaternions to rotation matrices
        pre_rot_mat = quaternion_to_rotation_matrix(pre_rot_quat)
        curr_rot_mat = quaternion_to_rotation_matrix(curr_rot_quat)

        # Compute the relative rotation and translation
        rot_mat_curr_pre = np.dot(curr_rot_mat, np.linalg.inv(pre_rot_mat))
        rot_mat_curr_pre = np.dot(np.linalg.inv(curr_rot_mat), pre_rot_mat)
        trans_vec_curr_pre = curr_trans_vec - np.dot(rot_mat_curr_pre, pre_trans_vec)
        # trans_vec_curr_pre = np.reshape(trans_vec_curr_pre, (3,1))

        return rot_mat_curr_pre, trans_vec_curr_pre


    # YWY comment: the reason we NOT use cv2.projectPoints->we want to keep depth of the point
    # while cv2.projectPoints() delete this info
    # def reproject_points(self, lidar_msg, im)
    def lidar_points_to_uvd(self, points_arr):
        rotated_points_arr = np.dot(self.R_cam_IMU, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_lidar_g
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]

        return uvd_points
    
    # 根据fruit_size 调整需要的knn中k的数量
    
    # fruit_point 包括 (x,y,size)
    # 根据 size 判断是否合理
    def find_fruit_depth(self, uvd_points, fruit_point):
        # Calculate the Manhattan distance between each point in uvd_points and fruit_point
        knn_points = self.find_knn(uvd_points, fruit_point)
        has_valid_depth = self.check_knn_dist(knn_points, fruit_point)
        if(not has_valid_depth):
            fruit_depth = False
        else:
            pass
        fruit_size = fruit_point.z
        fruit_depth = self.find_mode_depth(knn_points, fruit_size)
        # sol 1: 除法
        # if(fruit_depth*fruit_size>10 or fruit_depth/fruit_size<0.5):
        # sol 2: 乘法
        if(fruit_depth*fruit_depth*fruit_size>80):
            fruit_depth = False
        if(fruit_size>5 and fruit_depth>4):
            fruit_depth = False
        if(fruit_size<5 and fruit_depth<3):
            fruit_depth = False
        return fruit_depth
    
    def check_knn_dist(self, knn_points, fruit_point):
        distances = np.sqrt((knn_points[:, 0] - fruit_point.x)**2 + (knn_points[:, 1] - fruit_point.y)**2)
        if(distances[0] > 5):
            print(f"\ndistance[0] is {distances[0]}, larger than 5, False")
            return False
        else:
            return True


    def find_knn(self, uvd_points, fruit_point):
        # Calculate the Euclidean distance between each point in uvd_points and fruit_point
        distances = np.sqrt((uvd_points[:, 0] - fruit_point.x)**2 + (uvd_points[:, 1] - fruit_point.y)**2)
        fruit_size = int(fruit_point.z)
        print(f"KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN")
        print(f"fruit_size inside find_knn(fruit_point.z) is {fruit_size}")
        # Get the indices of the points sorted by their distances to fruit_point
        sorted_indices = np.argsort(distances)
        nearest_dist = distances[sorted_indices[0]]
        # nearest_dist_v = abs(uvd_points[sorted_indices[0],1]-fruit_point.y)
        # print(f"nearest_dist is {nearest_dist_v}")
        # if nearest_dist > 1:
        if nearest_dist > fruit_size*1.5:
            print("we have returned empty knn_points here")
            has_valid_depth = False
        else:
            has_valid_depth = True
        #     print(f"set has_valid_depth as {has_valid_depth}")
        # Select the top k points
        # 不同水果大小，需要的 knn 中 k 的数量根据 fruit_size 改变
        knn_indices = sorted_indices[:(int(fruit_size)+1)]
        knn_points = uvd_points[knn_indices]
        knn_points_dist = distances[knn_indices]
        # knn_points_in_fruit_size = []
        for i in (0,len(knn_points)-1):
            # pxiel_dist = abs(uvd_points)
            print(f"{i}th point is {knn_points[i]}, and pixel dist is {knn_points_dist[i]}")
        return knn_points

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
        z_values = uvd_points[:, 2]
        # Clip the z_values to the range [3, 8]
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
            cv2.circle(image, (x, y), 3, color=(int(b), int(g), int(r)), thickness=1)
        
        return image
        # Convert the OpenCV image to ROS Image message
        # colored_image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        # Publish the image
        # self.lidar_projected_image_pub.publish(colored_image_msg)
        # return colored_image_points
    
    def draw_bbx(self, image, x, y, size):
        # draw the fruit first
        image = cv2.circle(image, (int(x),int(y)), radius = size, color=(0,255,0), thickness=3)
        # Specify the center of the box (x, y)
        # Specify the width and height of the box
        w, h = 50, 30
        # Calculate the top left corner of the box
        x1, y1 = int(x - w/2), int(y - h/2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        # Add a text label with a background color
        label = "Yellow42"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x, text_y = x1, y1 - 10
        cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + text_size[1]), (0,255,0), -1)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0,0,0), thickness)
        return image
    
if __name__ == '__main__':
    try:
        lr = LidarReprojector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
