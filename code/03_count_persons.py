#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

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
cam_rgb.setFps(1)

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

# 2 more outputs (image and data)
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
    q_nn_right = device.getOutputQueue(name="right_nn", maxSize=4, blocking=False)
    q_nn_left = device.getOutputQueue(name="left_nn", maxSize=4, blocking=False)

    bboxes = []
    bboxes_r = []
    bboxes_l = []

    while True:
        in_nn = q_nn.get()
        in_nn_right = q_nn_right.get()
        in_nn_left = q_nn_left.get()
        
        if in_nn is not None:
            bboxes = in_nn.detections

        if in_nn_right is not None:
            bboxes_r = in_nn_right.detections

        if in_nn_left is not None:
            bboxes_l = in_nn_left.detections

        color = (255, 255, 255)

        if in_nn is not None:
            person_count = 0
            for bbox in bboxes:
                if bbox.label == 15:
                    person_count = person_count + 1
            print(f"center: {person_count} persons")

        if in_nn_right is not None:
            person_count = 0
            for bbox in bboxes_r:
                if bbox.label == 15:
                    person_count = person_count + 1
            print(f"right: {person_count} persons")

        if in_nn_left is not None:
            person_count = 0
            for bbox in bboxes_l:
                if bbox.label == 15:
                    person_count = person_count + 1
            print(f"left: {person_count} persons")

        if cv2.waitKey(1) == ord('q'):
            break
