import rospy
import cv2
import numpy as np
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import struct

# Camera intrinsics and T_cam_lidar
fx = 620.4325
fy = 640.4396
cx = 315.5857
cy = 225.9464
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
dist_coeffs = np.zeros((5, 1), np.float32)
R_cam_lidar_z_neg90 =  np.array([[0,1,0],
                                [-1, 0, 0],
                                [0,0.,1] ])
                                
# approximate
R_cam_lidar_y_neg104 =  np.array([[-0.2419219,  0.0000000, -0.9702957],
                                  [0.0000000,  1.0000000,  0.0000000],
                                  [0.9702957,  0.0000000, -0.2419219] ])

R_cam_lidar = np.dot(R_cam_lidar_z_neg90, R_cam_lidar_y_neg104)
rotvec_cam_lidar_g, _ = cv2.Rodrigues(R_cam_lidar)
transvec_cam_lidar_g = np.array([0.1,0,-0.1])

bridge = CvBridge()

# Create publisher
pub = rospy.Publisher('/fused_image', Image, queue_size=10)

def pointcloud2_to_xyz_array(cloud_msg):
    assert isinstance(cloud_msg, PointCloud2)
    
    # Create a generator for the points
    gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    
    # Create a list of the points
    points = list(gen)
    
    # Convert the list to a numpy array and return it
    return np.array(points)


# divide into
#   1. re-projection
#   2. colorize
def convert_and_colorize_points(homo_points):
    # Convert xy from homogeneous points to image points, keep z as its real dist
    image_points = homo_points[:, :2] / homo_points[:, 2, np.newaxis]
    z_values = homo_points[:, 2]
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
    colored_image_points = np.hstack((image_points, colors))

    return colored_image_points



def callback(lidar_msg, image):
    print("\ninside callback\n")
    # Convert ROS PointCloud2 to numpy array
    pc = pointcloud2_to_xyz_array(lidar_msg)
    # Remove points with x smaller than 0.3
    points = pc[pc[:, 0] >= 0.3]

    points_rotated = np.dot(R_cam_lidar, points.T)
    points_rotated_translated = points_rotated.T + transvec_cam_lidar_g
    homo_points = np.dot(camera_matrix, points_rotated_translated.T).T
    # homo_points[:, :2]: This is selecting all rows (indicated by :) and the first two columns (indicated by :2)
    # homo_points[:, 2, np.newaxis]: This is selecting all rows of the third column (indicated by 2) 
    # image_points = homo_points[:, :2] / homo_points[:, 2, np.newaxis]
    colored_image_points = convert_and_colorize_points(homo_points)

    # Convert ROS Image message to OpenCV image
    try:
        img = bridge.compressed_imgmsg_to_cv2(image, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    # Draw points on the image
    for x, y, b,g,r in np.int32(colored_image_points):
        # print(f"b,g,r:{b},{g},{r}")
        cv2.circle(img, (x, y), 3, color=(int(b), int(g), int(r)), thickness=3)
    
    # Convert the OpenCV image to ROS Image message
    img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
    
    # Publish the image
    pub.publish(img_msg)

def listener():
    rospy.init_node('listener', anonymous=True)
    
    # Subscribe to both topics
    sub_lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
    sub_image = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
    
    # Synchronize the topics with a slop of 0.1 seconds
    ts = message_filters.ApproximateTimeSynchronizer([sub_lidar, sub_image], 10, 0.1)
    ts.registerCallback(callback)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
