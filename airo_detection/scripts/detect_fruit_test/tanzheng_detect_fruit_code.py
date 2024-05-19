import cv2
import numpy as np
 
# it feels like hsv is NOT as good as RGB

# Set up the detector with default parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
# params.minArea = 20  # adjust this value depending on the size of the blobs you want to detect
params.filterByColor = False
params.minThreshold = 65
params.maxThreshold = 93
params.blobColor = 0
params.minArea = 10
params.maxArea = 90
params.filterByCircularity = False
params.filterByConvexity = False
params.minCircularity =.4
params.maxCircularity = 1
detector = cv2.SimpleBlobDetector_create(params)



def nothing(x):
    pass

# Mouse callback function
def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y,x]
        print('HSV Value at ({},{}): {}'.format(x, y, pixel))

        # Draw a point at the clicked position
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Display the HSV value near the point
        cv2.putText(image, str(pixel), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # 1.Load the image
# image = cv2.imread('/home/airo_ws/icuas_final_bags/hsv_test_images/test1.png')

# # 2.Convert the image to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('HSV Tuner')

# Set mouse callback function to window
cv2.setMouseCallback('HSV Tuner', get_hsv)

# Create trackbars for color change
cv2.createTrackbar('LowerH', 'HSV Tuner', 20, 179, nothing)
cv2.createTrackbar('UpperH', 'HSV Tuner', 67, 179, nothing)
cv2.createTrackbar('LowerS', 'HSV Tuner', 26, 255, nothing)
cv2.createTrackbar('UpperS', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('LowerV', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('UpperV', 'HSV Tuner', 255, 255, nothing)

while(1):
    # 1. Load the image inside the loop
    image = cv2.imread('/home/airo_ws/icuas_final_bags/hsv_test_images/test9.png')
    # 2. erosion
    kernel = np.ones((3,3),np.uint8)
    cv2.erode(image,kernel,iterations=1)

    # 3. expand iteration 1st enough
    dige_dilate= cv2.dilate(image,kernel,iterations=1)
    dilate1 = cv2.dilate(image,kernel,iterations=1)
    # dilate2 = cv2.dilate(image,kernel,iterations=2)
    # dilate3 = cv2.dilate(image,kernel,iterations=3)

    # 4. Convert the image to HSV color space
    hsv_no_clahe = cv2.cvtColor(dilate1, cv2.COLOR_BGR2HSV)
    # Split the image into its respective channels
    h, s, v = cv2.split(hsv_no_clahe)
    # 5. Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Apply CLAHE to each channel
    h_clahe = clahe.apply(h)
    s_clahe = clahe.apply(s)
    v_clahe = clahe.apply(v)

    # Merge the CLAHE enhanced channels back into an image
    # this image is actually image_clahe
    hsv = cv2.merge([h_clahe, s_clahe, v_clahe])

    # 4. Get current positions of the trackbars
    lower_yellow = np.array([cv2.getTrackbarPos('LowerH', 'HSV Tuner'), cv2.getTrackbarPos('LowerS', 'HSV Tuner'), cv2.getTrackbarPos('LowerV', 'HSV Tuner')])
    upper_yellow = np.array([cv2.getTrackbarPos('UpperH', 'HSV Tuner'), cv2.getTrackbarPos('UpperS', 'HSV Tuner'), cv2.getTrackbarPos('UpperV', 'HSV Tuner')])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # 4. erosion and expan
    # kernel = np.ones((3,3),np.uint8)
    # dige_erosion = cv2.erode(image,kernel,iterations=1)
    # # 膨胀操作
    # dige_dilate= cv2.dilate(image,kernel,iterations=1)
    # # show_img.show_img('dilate',dige_dilate)
    # dilate1 = cv2.dilate(image,kernel,iterations=1)
    # dilate2 = cv2.dilate(image,kernel,iterations=2)
    # dilate3 = cv2.dilate(image,kernel,iterations=3)
    # res = np.hstack((image,dige_erosion,dilate1,dilate2,dilate3))
    # 4. Bitwise-AND mask and original image
    res = cv2.bitwise_and(dilate1, dilate1, mask=mask)
    # Display the image with contours
    # 5. Use the blob detector to find blobs in the mask
    keypoints = detector.detect(res)
    print("this is good, man\n")
    for keypoint in keypoints:
        print(f"keypoint is {keypoint.pt}\n")
        print(f"diameter is are {keypoint.size}\n")

    # Draw detected blobs as red circles
    im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('HSV with blob', im_with_keypoints)


    cv2.imshow('HSV Tuner', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()