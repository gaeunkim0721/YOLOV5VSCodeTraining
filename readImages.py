import cv2
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from utils.general import (CONFIG_DIR, FONT, LOGGER, check_font, check_requirements, clip_coords, increment_path,
                           is_ascii, xywh2xyxy, xyxy2xywh)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(r'C:\Users\Gaeun\Desktop\Gaeun\walking.mp4')

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')# or yolov5n - yolov5x6, custom
model.to(device)

model.conf = 0.75  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes =  [0]  # 
 
 
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
    
lst = []
blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255) 
# 폰트 지정
font =  cv2.FONT_HERSHEY_PLAIN

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    frame_count = 0
    t0 = datetime.now()
    ret, frame = cap.read()
    frame_count += 1
    frame = cv2.resize(frame, (700, 400))
    if ret == True:
        # Inference
        results = model(frame)
        avg_fps = frame_count / (datetime.now()-t0).microseconds*1000000
        inferenceT = 1/avg_fps*1000
        avg_fps = "FPS" + str(round(avg_fps,2))
        inferenceT = "Time: " + str(round(inferenceT,2)) + "ms"
            
        cv2.putText(frame, avg_fps, (50, 20), font, 1, red, 2, cv2.LINE_AA)
        cv2.putText(frame, inferenceT, (500, 20), font, 1, white, 2, cv2.LINE_AA)
        cv2.imshow('YOLOV5', np.squeeze(results.render()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()