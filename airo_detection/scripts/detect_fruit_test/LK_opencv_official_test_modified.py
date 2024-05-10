import numpy as np
import cv2
import argparse
import glob
import sys, os
from cv_bridge import CvBridge, CvBridgeError
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#  The example file can be downloaded from: \
#  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# cap = cv2.VideoCapture(args.image)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
 qualityLevel = 0.3,
 minDistance = 7,
 blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
 maxLevel = 2,
 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
# Function to extract the numeric part of the filename

tracking_test_path = os.path.dirname(__file__)
sys.path.append(tracking_test_path)
data_dir = os.path.join(tracking_test_path,"tracking_test_data")
print(data_dir)
file_list = glob.glob(os.path.join(data_dir,"*.png"))
# Sort the files by timestamp (assuming filenames are timestamps)
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
old_frame = cv2.imread(file_list[0])
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
for i in range(1,len(file_list)):
    frame = cv2.imread(file_list[i])
    #  if not ret:
    #  print('No frames grabbed!')
    #  break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()