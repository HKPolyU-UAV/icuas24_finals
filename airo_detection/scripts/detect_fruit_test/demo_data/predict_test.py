from ultralytics import YOLO
import sys, os
import cv2
import numpy as np
yolo_test_path = os.path.dirname(__file__)
sys.path.append(yolo_test_path)
# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")
print(f"before we load the model and the image\n\n")
# Load a pretrained YOLO model (recommended for training)
model = YOLO("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/last.pt")


# Load the image
image = cv2.imread("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/red_yellow.png")

# Perform object detection on the image using the model
results = model(image)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    class_indices = boxes.cls  # Class indices of each detected object
    confidences = boxes.conf  # Confidence scores of each detected object

    xywhs = boxes.xywh

    
    for xywh, class_index, confidence  in zip(xywhs, class_indices, confidences):
        class_name = result.names[int(class_index)]
        print(f"xywh: {xywh}")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.2f}")



    print(f"=======================================================")
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

