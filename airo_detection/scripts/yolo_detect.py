from ultralytics import YOLO
import sys, os
import cv2
import numpy as np
from geometry_msgs.msg import Point

yolo_test_path = os.path.dirname(__file__)
sys.path.append(yolo_test_path)
# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")
print(f"before we load the model and the image\n\n")
# Load a pretrained YOLO model (recommended for training)
# Load a pretrained YOLO model (recommended for training)/home/allen/icuas24_ws_mini/src/airo_detection/scripts/last_yolo.pt

# model = YOLO("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/last.pt")


# Load the image
# image = cv2.imread("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/red_yellow.png")

# Perform object detection on the image using the model

def yolo_detect(image, model):
    print("yolo_detect() called")
    # Process results list
    yolo_fruit_points = []
    results = model(image)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        class_indices = boxes.cls  # Class indices of each detected object
        confidences = boxes.conf  # Confidence scores of each detected object

        xywhs = boxes.xywh

        
        for xywh, class_index, confidence  in zip(xywhs, class_indices, confidences):
            class_name = result.names[int(class_index)]
            print(f"xywh: {xywh}")
            print(f"Class: {class_name}")
            print(f"class index is: {class_index}")
            print(f"Confidence: {confidence:.2f}")
            if(class_name == "yellow"):
                print("inside")
                point = Point()
                point.x = float(xywh[0])
                point.y = float(xywh[1])
                point.z = float((xywh[2]+xywh[3])/2)
                yolo_fruit_points.append(point)
                
    return yolo_fruit_points
    # print(f"=======================================================")
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk

