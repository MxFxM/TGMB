#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import glob
import os
from lxml import etree as ET
from tqdm import tqdm

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Input stream
frame_in = pipeline.createXLinkIn()
frame_in.setStreamName("frame_input")

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setConfidenceThreshold(0.6)
detection_nn.setBlobPath('models/bee_detection_v2021_202104141915.blob')
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)
frame_in.out.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_img_in = device.getInputQueue(name="frame_input", maxSize=1, blocking=True)
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    #for nextfile in tqdm(glob.glob("images/20210530150605")):
    for nextfile in tqdm(glob.glob("images/*")):
        name = nextfile
        # load image into frame
        frame = cv2.imread(nextfile, cv2.IMREAD_COLOR)
        original_frame = frame.copy()

        # instead of resizing, crop the image into multiple smaller pieces
        # then process each 300x300 piece
        # and put all the results back together

        # 300, 600, 900, 1200, 1500, 1800
        # slice 1920 into 6 pieces
        # slice 1080 into 3 pieces
        # resulting in 18 pieces per frame
        # this is slow btw, because each frame takes 18 times as long
        for x_start in [0, 300, 600, 900, 1200, 1500]:
            for y_start in [0, 300, 600]:
                frame = original_frame[y_start:y_start+300, x_start:x_start+300]

                var_data = dai.NNData()
                var_data.setLayer("data", to_planar(frame, (300,300)))
                q_img_in.send(var_data)

                in_nn = q_nn.get()
                detections = in_nn.detections

                for detection in detections:
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    
                    #det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    x1 = bbox[0] + x_start
                    x2 = bbox[2] + x_start
                    y1 = bbox[1] + y_start
                    y2 = bbox[3] + y_start

                    cv2.rectangle(original_frame, (x1,y1), (x2,y2), (255, 0, 0), 2)

        #cv2.imshow(name, original_frame)
        cv2.imwrite(f"./images/labeled/{name[7:]}_det.png", original_frame)

        #cv2.waitKey()

        #break
