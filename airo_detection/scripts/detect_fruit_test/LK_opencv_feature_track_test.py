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
lk_params = dict(winSize=(100, 100),
                 maxLevel=5,
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


# Read the first image
old_frame = cv2.imread(file_list[0])
old_frame = twoD_fruit_detector.detect_fruit_only_mask(old_frame)
old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# p0 = cv2.goodFeaturesToTrack(old_frame_gray, mask=None, **feature_params)
p0 = twoD_fruit_detector.detect_fruit(old_frame)

# Draw the features
for pt in p0:
    cv2.circle(old_frame, tuple(pt[0]), 1, (0, 0, 255), 2)

cv2.imshow("Feature Image", old_frame)

keypoints_history = []
# Create an empty image to draw the history of keypoints
history_image = np.zeros_like(old_frame)

# ================================
# setup initial location of window
x, y, w, h = 370, 375, 10, 10 # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
# old_hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
# old_clahed_hsv = twoD_fruit_detector.clahe_image(old_hsv)
# old_threshold_clahe = twoD_fruit_detector.clahed_hsv_fruit_mask(old_clahed_hsv)
# hsv_roi = old_threshold_clahe[y:y+h, x:x+w]
# # hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# mask = old_threshold_clahe
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,255.)), np.array((80.,255.,255.)))
# cv2.imshow("mask this is", mask)
# print(f"here we are at the end of mask")
# cv2.waitKey()
# # img2  = cv2.circle(mask, (int(x),int(y)), radius = 5, color=(0,255,0), thickness=2)
# cv2.imshow("circled mask this is", img2  )
# cv2.waitKey()
# roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0 )
# ===========================
cv2.waitKey()
# For each subsequent image...
for i in range(1, len(file_list)):
    # Read the image
    new_frame = cv2.imread(file_list[i])
    cv2.imshow("rgb frame", new_frame)
    new_frame = twoD_fruit_detector.detect_fruit_only_mask(new_frame)
    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    # p1 = cv2.goodFeaturesToTrack(new_frame_gray, mask=None, **feature_params)
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, new_frame_gray, p0, p1, **lk_params)
    # good_new = p1[st == 1]
    # good_old = p0[st == 1]

    p1 = twoD_fruit_detector.detect_fruit(new_frame)
    for fruit in p1:
        frame = cv2.circle(new_frame, (int(fruit[0][0]),int(fruit[0][1])), radius = 5, color=(0,255,0), thickness=2)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray, new_frame_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]



    # Add the new keypoints to the history
    keypoints_history.append(good_new)

    # Draw the features and the flow vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(new_frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(new_frame, (a, b), 5, color[i].tolist(), -1)

    # Draw the history of keypoints
    for pts in keypoints_history:
        for pt in pts:
            x, y = pt.ravel()
            new_frame = cv2.circle(new_frame, (x, y), 3, color[i].tolist(), -1)

    # cv2.imshow('new_frame', new_frame)
    # cv2.imshow('history', history_image)  # Show the history image
    # cv2.waitKey(100)

    # Swap old and new points and images
    p0, p1 = p1, p0
    old_frame_gray, new_frame_gray = new_frame_gray, old_frame_gray
    # Count the number of tracked points
    # tr_num = np.count_nonzero(status)

    # If too few points are being tracked, detect new features
    # if tr_num < 6:
    #     print("You need to change the feat-img because the background-img was all changed")
    #     point1 = cv2.goodFeaturesToTrack(image2_gray, mask=None, **feature_params)
    #     pointCopy = point1.copy()

    # Draw the features and the flow vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        new_frame = cv2.line(new_frame, (a, b), (c, d), color[i].tolist(), 2)
        new_frame = cv2.circle(new_frame, (a, b), 5, color[i].tolist(), -1)

    # hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    # clahed_hsv = twoD_fruit_detector.clahe_image(hsv)
    # threshold_clahe = twoD_fruit_detector.clahed_hsv_fruit_mask(clahed_hsv)
    # dst = cv2.calcBackProject([threshold_clahe],[0],roi_hist,[0,180],1)

    # apply camshift to get the new location
    # ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    # pts = cv2.boxPoints(ret)
    # pts = np.int0(pts)
    # img2 = cv2.polylines(frame,[pts],True, 255,2)
    # clahed_img2 = cv2.polylines(threshold_clahe,[pts],True, 255,2)
    # cv2.imshow('img2',img2)
    # cv2.imshow('clahed_img2',clahed_img2)


    cv2.imshow('new_frame', new_frame)
    cv2.waitKey(100)
    # Swap old and new points and images
    # point1, point2 = point2, point1
    # image1, image2 = image2, image1

cv2.destroyAllWindows()
