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

import sys, os
airo_detection_path = os.path.dirname(__file__)
sys.path.append(airo_detection_path)
from threeD_fruit_manager import *
from transform_utils import *

class LidarReprojector:
    def __init__(self):
        # Camera intrinsics and T_cam_lidar
        self.bridge = CvBridge()
        self.fx = 620.4325
        self.fy = 640.4396
        self.cx = 315.5857
        self.cy = 225.9464
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)
        self.dist_coeffs = np.zeros((5, 1), np.float32)
        self.R_cam_lidar_z_neg90 =  np.array([[0,1,0],
                                [-1, 0, 0],
                                [0,0.,1] ])
                                
        # approximate
        self.R_cam_lidar_y_neg104 =  np.array([[-0.2419219,  0.0000000, -0.9702957],
                                        [0.0000000,  1.0000000,  0.0000000],
                                        [0.9702957,  0.0000000, -0.2419219] ])

        self.R_cam_lidar = np.dot(self.R_cam_lidar_z_neg90, self.R_cam_lidar_y_neg104)
        # self.R_cam_lidar = np.dot(np.array([[0,1,0], [-1, 0, 0], [0,0.,1]]), np.array([[-0.2419219,  0.0000000, -0.9702957], [0.0000000,  1.0000000,  0.0000000], [0.9702957,  0.0000000, -0.2419219]]))
        self.rotvec_cam_lidar_g, _ = cv2.Rodrigues(self.R_cam_lidar)
        self.transvec_cam_lidar_g = np.array([0.1,0,-0.1])
        self.lidar_projected_image_pub = rospy.Publisher('/fused_image_ooad', Image, queue_size=10)

        self.fruit_database = PlantFruitDatabase()
        self.transform_utils = TransformUtils()

        # ROS node and subscriber
        rospy.init_node('lidar_reprojector', anonymous=True)
        # rospy.Subscriber("/velodyne_points", PointCloud2, self.callback)

        # Subscribe to both topics
        self.sub_lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        self.sub_image = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        self.sub_2dfruit = message_filters.Subscriber('two_d_fruit_keypoints', PointStamped)
        # self.sub_odom = message_filters.Subscriber('/kiss/odometry', Odometry)
        self.sub_odom = message_filters.Subscriber('/Odometry', Odometry)

        # Synchronize the topics with a slop of 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_lidar, self.sub_image, self.sub_2dfruit, self.sub_odom], 10, 0.0112)
        self.ts.registerCallback(self.callback)


    def callback(self, lidar_msg, image_msg, two_d_fruit_keypoints_msg, odom_msg):
        # Convert ROS PointCloud2 to PCL
        points_arr = self.lidar_msg_to_xyz_array(lidar_msg)

        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Reproject points
        points_arr = self.lidar_msg_to_xyz_array(lidar_msg)
        uvd_points = self.lidar_points_to_uvd(points_arr)  # YWY noly need uvd
        fruit_point = Point()
        fruit_point = two_d_fruit_keypoints_msg.point
        image = self.draw_bbx(image, int(fruit_point.x),int(fruit_point.y))
        image_with_lidar_points = self.colorize_uvd_visualization(uvd_points, image)  #colored_image_points only for visualization, could delete

        # fruit_depth = self.find_fruit_depth(uvd_points, fruit_point)
        # XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, fruit_depth, odom_msg)

        knn_points = self.find_knn(uvd_points, fruit_point, 5)
        mode_depth = self.find_mode_depth(knn_points)
        fruit_depth = mode_depth
        print(f"MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH_MODE_DEPTH={mode_depth}")
        XYZ_yellow = self.transform_utils.uvd_to_world(fruit_point.x, fruit_point.y, mode_depth, odom_msg)
        
        # if(fruit_depth > 3 and fruit_depth < 9 and XYZ_yellow[0] > 2 and XYZ_yellow[0]< 8) and XYZ_yellow[2]<0.5 and XYZ_yellow[2]>-2:
        if(fruit_depth > 2 and fruit_depth < 7.5):
            curr_yellow_id = len(self.fruit_database.fruit_arr_.markers)
            print(f"we have============================= {curr_yellow_id} ============================yellow fruits now")
            # self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(rpy_ywy[0]))
            self.fruit_database.add_fruit_marker('yellow', curr_yellow_id, XYZ_yellow, abs(8))
            self.fruit_database.publish_markers()
            colored_image_msg = self.bridge.cv2_to_imgmsg(image_with_lidar_points, "bgr8")
            colored_image_msg.header = odom_msg.header
            self.lidar_projected_image_pub.publish(colored_image_msg)
            print("BINGO\n")
            # print(f"fruit depth is:{fruit_depth}")

        # image.clear()
        # uvd_points.clear()

    def lidar_msg_to_xyz_array(self, lidar_msg):
        assert isinstance(lidar_msg, PointCloud2)
        
        # Create a generator for the points
        point = point_cloud2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        
        # Create a list of the points
        points_arr = list(point)
        
        # Convert the list to a numpy array and return it
        points_arr = np.array(points_arr)
        points_arr = points_arr[points_arr[:, 0] >= 1]
        points_arr = points_arr[points_arr[:, 1] <10]
        points_arr = points_arr[-10< points_arr[:, 1]]

        return np.array(points_arr)
    
    # YWY comment: the reason we NOT use cv2.projectPoints->we want to keep depth of the point
    # while cv2.projectPoints() delete this info
    # def reproject_points(self, lidar_msg, im)
    def lidar_points_to_uvd(self, points_arr):
        rotated_points_arr = np.dot(self.R_cam_lidar, points_arr.T)
        translated_rotated_points_arr = rotated_points_arr.T + self.transvec_cam_lidar_g
        homo_points = np.dot(self.camera_matrix, translated_rotated_points_arr.T).T
        # image_points -> (x,y,depth)
        uvd_points = homo_points[:, :3] / homo_points[:, 2, np.newaxis]
        uvd_points[:,2] = homo_points[:,2]

        return uvd_points
    
    def find_fruit_depth(self, uvd_points, fruit_point):
        # Calculate the Manhattan distance between each point in uvd_points and fruit_point
        distances = np.abs(uvd_points[:, 0] - fruit_point.x) + np.abs(uvd_points[:, 1] - fruit_point.y)
        # Find the indices of points where distance < 4
        indices = np.where(distances < 5)

        # Select these points
        selected_points = uvd_points[indices]

        # If no points satisfy the condition, return False
        if selected_points.size == 0:
            # print("False")
            return False
        else:
            # Find the point with the minimum distance
            min_index = np.argmin(distances[indices])
            min_point = selected_points[min_index]
            fruit_depth = min_point[2]
            # if(fruit_depth)
            # print("Minimum distance point depth: ", min_point)
            return fruit_depth
    
    from scipy.spatial import distance

    def find_knn(self, uvd_points, fruit_point, k):
        # Calculate the Euclidean distance between each point in uvd_points and fruit_point
        distances = np.sqrt((uvd_points[:, 0] - fruit_point.x)**2 + (uvd_points[:, 1] - fruit_point.y)**2)
        
        # Get the indices of the points sorted by their distances to fruit_point
        sorted_indices = np.argsort(distances)
        
        # Select the top k points
        knn_indices = sorted_indices[:k]
        knn_points = uvd_points[knn_indices]
        print(f"KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN_KNN")
        for k_ponit in knn_points:
            print(f"the depth of kth points is {k_ponit[2]}")
        return knn_points

    def find_mode_depth(self, knn_points):
        # Extract the depth values
        depths = knn_points[:, 2]

        # Round the depths to one decimal place
        rounded_depths = np.round(depths, 1)

        # Calculate the mode
        mode_depth = statistics.mode(rounded_depths)
        # Calculate the mode
        mode_depth = statistics.mode(rounded_depths)

        # Count the occurrences of the mode
        mode_count = np.count_nonzero(rounded_depths == mode_depth)

        # Calculate 80% of the number of points
        eighty_percent_count = 0.6 * len(depths)
        # Check if the mode count is greater than or equal to 80% of the number of points
        if mode_count >= eighty_percent_count:
            print(f"mode depth larger than 60%, mode is {mode_depth}")
            return mode_depth
        else:
            print(f"mode depth less than 60%, discard")
            return False

    def colorize_uvd_visualization(self, uvd_points, image):
        # Convert xy from homogeneous points to image points, keep z as its real dist
        z_values = uvd_points[:, 2]
        # Clip the z_values to the range [3, 8]
        z_values = np.clip(z_values, 3, 8)
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
    
    def draw_bbx(self, image, x, y):
        # draw the fruit first
        image = cv2.circle(image, (int(x),int(y)), radius = 3, color=(0,255,0), thickness=3)
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
