#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

'''
Mobilenet SSD device side decoding demo
  The "mobilenet-ssd" model is a Single-Shot multibox Detection (SSD) network intended
  to perform object detection. This model is implemented using the Caffe* framework.
  For details about this model, check out the repository <https://github.com/chuanqi305/MobileNet-SSD>.
'''

# MobilenetSSD label texts
label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Get argument first
mobilenet_path = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(20)

# Define a neural network that will make predictions based on the source frames
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setBlobPath(mobilenet_path)
#detectionNetwork.setNumInferenceThreads(2) # limit inference to run multiple networks simultaneously
detectionNetwork.input.setBlocking(False)
cam_rgb.preview.link(detectionNetwork.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
detectionNetwork.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("rgb_nn")
detectionNetwork.out.link(xout_nn.input)

# Define 2 more sources
cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# resize the mono images to 300x300 for the nn
manip_right = pipeline.createImageManip()
manip_right.initialConfig.setResize(300, 300)
manip_right.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
cam_right.out.link(manip_right.inputImage)

manip_left = pipeline.createImageManip()
manip_left.initialConfig.setResize(300, 300)
manip_left.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
cam_left.out.link(manip_left.inputImage)

# 2 more networks
detection_right = pipeline.createMobileNetDetectionNetwork()
detection_right.setConfidenceThreshold(0.5)
detection_right.setBlobPath(mobilenet_path)
detection_right.input.setBlocking(False)
manip_right.out.link(detection_right.input)

detection_left = pipeline.createMobileNetDetectionNetwork()
detection_left.setConfidenceThreshold(0.5)
detection_left.setBlobPath(mobilenet_path)
detection_left.input.setBlocking(False)
manip_left.out.link(detection_left.input)

# 4 more ouputs (image and data)
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")
detection_right.passthrough.link(xout_right.input)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName("left")
detection_left.passthrough.link(xout_left.input)

xout_nn_right = pipeline.createXLinkOut()
xout_nn_right.setStreamName("right_nn")
detection_right.out.link(xout_nn_right.input)

xout_nn_left = pipeline.createXLinkOut()
xout_nn_left.setStreamName("left_nn")
detection_left.out.link(xout_nn_left.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="rgb_nn", maxSize=4, blocking=False)

    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    q_nn_right = device.getOutputQueue(name="right_nn", maxSize=4, blocking=False)

    q_left = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    q_nn_left = device.getOutputQueue(name="left_nn", maxSize=4, blocking=False)

    frame = None
    frame_r = None
    frame_l = None
    bboxes = []
    bboxes_r = []
    bboxes_l = []

    while True:
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()

        in_right = q_right.get()
        in_nn_right = q_nn_right.get()

        in_left = q_left.get()
        in_nn_left = q_nn_left.get()
        
        if in_rgb is not None:
            shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
            frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if in_nn is not None:
            bboxes = in_nn.detections

        if in_right is not None:
            shape = (3, in_right.getHeight(), in_right.getWidth())
            frame_r = in_right.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame_r = np.ascontiguousarray(frame_r)

        if in_nn_right is not None:
            bboxes_r = in_nn_right.detections

        if in_left is not None:
            shape = (3, in_left.getHeight(), in_left.getWidth())
            frame_l = in_left.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame_l = np.ascontiguousarray(frame_l)

        if in_nn_left is not None:
            bboxes_l = in_nn_left.detections

        color = (255, 255, 255)

        if frame is not None:
            height = frame.shape[0]
            width  = frame.shape[1]
            for bbox in bboxes:
                x1 = int(bbox.xmin * width)
                x2 = int(bbox.xmax * width)
                y1 = int(bbox.ymin * height)
                y2 = int(bbox.ymax * height)
                try:
                    label = label_map[bbox.label]
                except:
                    label = bbox.label
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                #cv2.putText(frame, "{:.2f}".format(bbox.confidence*100), (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imshow("rgb", frame)

        if frame_r is not None:
            height = frame_r.shape[0]
            width  = frame_r.shape[1]
            for bbox in bboxes_r:
                x1 = int(bbox.xmin * width)
                x2 = int(bbox.xmax * width)
                y1 = int(bbox.ymin * height)
                y2 = int(bbox.ymax * height)
                try:
                    label = label_map[bbox.label]
                except:
                    label = bbox.label
                cv2.putText(frame_r, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame_r, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imshow("right", frame_r)

        if frame_l is not None:
            height = frame_l.shape[0]
            width  = frame_l.shape[1]
            for bbox in bboxes_l:
                x1 = int(bbox.xmin * width)
                x2 = int(bbox.xmax * width)
                y1 = int(bbox.ymin * height)
                y2 = int(bbox.ymax * height)
                try:
                    label = label_map[bbox.label]
                except:
                    label = bbox.label
                cv2.putText(frame_l, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame_l, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.imshow("left", frame_l)

        if cv2.waitKey(1) == ord('q'):
            break
