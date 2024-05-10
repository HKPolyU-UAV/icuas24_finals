import cv2
import numpy as np
import glob
import sys, os
from cv_bridge import CvBridge, CvBridgeError

tracking_test_path = os.path.dirname(__file__)
sys.path.append(tracking_test_path)
# Get the parent directory
parent_dir = os.path.dirname(tracking_test_path)
# Add the parent directory to the system path
sys.path.append(parent_dir)

from twoDim_fruit_detector import TwoDFruitDetector

twoD_fruit_detector = TwoDFruitDetector()

data_dir = os.path.join(tracking_test_path,"tracking_test_data")
print(data_dir)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=10,
                      blockSize=3)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(50, 50),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Get list of all .png files in the specified directory
# file_list = glob.glob('/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/tracking_test_data/*.png')

file_list = glob.glob(os.path.join(data_dir,"*.png"))
# Sort the files by timestamp (assuming filenames are timestamps)

# Function to extract the numeric part of the filename
def extract_number(filename):
    # Remove the directory and extension, leaving only the filename
    base_name = os.path.basename(filename)
    name_without_extension = os.path.splitext(base_name)[0]
    # Extract the number from the filename
    number = float(name_without_extension)
    return number

# Sort the files by the numeric part of the filename
file_list.sort(key=extract_number)
# file_list.sort(key=os.path.getmtime)
print(f"file_list[0]: {file_list[0]}")
# Take first frame and find corners in it
old_frame = cv2.imread(file_list[0])
bridge = CvBridge()
# old_frame_msg = bridge.cv2_to_imgmsg(old_frame, "bgr8")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p_fruit0 = twoD_fruit_detector.detect_fruit(old_frame)

# Iterate over each file in the directory
for i in range(1, len(file_list)):
    frame = cv2.imread(file_list[i])
    print(f"file_list[{i}]: {file_list[i]}")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    # print(f"p1: {p1}")
    p_fruit1 = twoD_fruit_detector.detect_fruit(frame)
    # for fruit in p_fruit1:
    #     frame = cv2.circle(frame, (int(fruit[0][0]),int(fruit[0][1])), radius = 3, color=(0,255,255), thickness=1)

    # print(f"p_fruit1.shape is {p_fruit1.shape}")
    # print(f"p_fruit1 is {p_fruit1}")
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # p_fruit1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_fruit0, p_fruit1, **lk_params)
    # good_new = p_fruit1[st == 1]
    # good_old = p_fruit0[st == 1]

    # for fruit in p_fruit1:
    #     frame = cv2.circle(frame, (int(fruit[0][0]),int(fruit[0][1])), radius = 2, color=(0,255,0), thickness=3)

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    cv2.imshow('Frame', frame)
    cv2.waitKey(1000)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
