import numpy as np
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, PointCloud2, CompressedImage        


# @TODO we will need to track 2D fruits in the image plane
# @TODO then generate stable 3D fruit by depth
# @TODO
# @TODO
# @TODO

class TwoDFruitDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("camera/color/image_raw/compressed", CompressedImage, self.callback)
        # self.keypoints_pub = rospy.Publisher("two_d_fruit_keypoints", PointStamped, queue_size=10)
        # self.annoted_image_pub = rospy.Publisher('/fused_image', Image, queue_size=10)
        self.fruit_points_ = []

    def detect_fruit_only_mask(self, image):
        self.fruit_points_.clear()
        # try:
        #     cv_image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        # except cv2.CvBridgeError as e:
        #     print(e)

        # 1. RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 2. HSV to CLAHED_HSV
        clahed_hsv_image = self.clahe_image(hsv_image)
        # 3. mask by lower&upper HSV
        mask = self.clahed_hsv_fruit_mask(clahed_hsv_image)
        return mask

    def detect_fruit(self, image):
        self.fruit_points_.clear()
        # try:
        #     cv_image = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")
        # except cv2.CvBridgeError as e:
        #     print(e)

        # 1. RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 2. HSV to CLAHED_HSV
        clahed_hsv_image = self.clahe_image(hsv_image)
        # 3. mask by lower&upper HSV
        mask = self.clahed_hsv_fruit_mask(clahed_hsv_image)
        # 4. Blob detector
        keypoints = self.blob_detector(mask)
        # 5. pub 2D fruits
        # detected_points_arr = PointStamped()
        # detected_points_arr.header = image_msg.header
        for keypoint in keypoints:
            point = Point()
            if(abs(keypoint.pt[0]-320) < 150 and 100 < keypoint.pt[1] < 380):
                point.x = keypoint.pt[0]
                point.y = keypoint.pt[1]
                point.z = keypoint.size
                self.fruit_points_.append(point)
                # detected_points_arr.points.append(point)
                # point.header = image_msg.header
                # self.keypoints_pub.publish(point)
                print(f"we have pub a 2d fruit at image center the fruit image coor is ({keypoint.pt[0]},{keypoint.pt[1]})")
            else:
                print(f"this fruit at({keypoint.pt[0]},{keypoint.pt[1]}) is NOT qualified")
        # Convert keypoints to the same format as goodFeaturesToTrack
        keypoints_goodFeaturesToTrack_format = np.array([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
        keypoints_goodFeaturesToTrack_format_float32 = np.float32(np.round(keypoints_goodFeaturesToTrack_format))
        print(keypoints_goodFeaturesToTrack_format_float32)
        return keypoints_goodFeaturesToTrack_format_float32
        # self.keypoints_pub.publish(str(keypoints))


    def clahe_image(self, hsv_image_noclahe):
        # Split the image into its respective channels
        h, s, v = cv2.split(hsv_image_noclahe)
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Apply CLAHE to each channel
        h_clahe = clahe.apply(h)
        s_clahe = clahe.apply(s)
        v_clahe = clahe.apply(v)
        # Merge the CLAHE enhanced channels back into an image
        # this image is actually image_clahe
        clahed_hsv_image = cv2.merge([h_clahe, s_clahe, v_clahe])
        return clahed_hsv_image
    
    def clahed_hsv_fruit_mask(self, clahed_hsv_image):
        # manually tuned by YWY
        lower_yellow = np.array([0, 60, 255])
        upper_yellow = np.array([80, 255, 255])
        mask = cv2.inRange(clahed_hsv_image, lower_yellow, upper_yellow)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(clahed_hsv_image, clahed_hsv_image, mask=mask)
        return res

    def blob_detector(self, res):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        # params.minArea = 20  # adjust this value depending on the size of the blobs you want to detect
        params.filterByColor = False
        params.minThreshold = 65
        params.maxThreshold = 93
        params.blobColor = 0
        params.minArea = 8
        params.maxArea = 90
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.minCircularity =.4
        params.maxCircularity = 1
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(res)
        # Draw detected blobs as red circles
        # image_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # for keypoint in keypoints:
            # image_with_keypoints = cv2.circle(image_with_keypoints, (int(keypoint.pt[0]),int(keypoint.pt[1])), radius = 3, color=(0,255,0), thickness=3)
        # Convert the OpenCV image to ROS Image message
        # img_msg = self.bridge.cv2_to_imgmsg(image_with_keypoints, "bgr8")
        
        # Publish the image
        # self.annoted_image_pub.publish(img_msg)

        return keypoints




