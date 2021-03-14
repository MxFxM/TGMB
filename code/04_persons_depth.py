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
detectionNetwork.setConfidenceThreshold(0.8) # increased threshold, because there is no threshold later on
detectionNetwork.setBlobPath(mobilenet_path)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)
cam_rgb.preview.link(detectionNetwork.input)

# Create outputs
#xout_rgb = pipeline.createXLinkOut()
#xout_rgb.setStreamName("rgb")
#detectionNetwork.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("rgb_nn")
detectionNetwork.out.link(xout_nn.input)

# Define 2 more sources
cam_right = pipeline.createMonoCamera()
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # reduced resolution

cam_left = pipeline.createMonoCamera()
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# depth by disparity
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True) # mirror image
depth.setRectifyEdgeFillColor(0) # black on the edges
cam_left.out.link(depth.left)
cam_right.out.link(depth.right)

# depth output
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.disparity.link(xout_depth.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames, depth info and nn data from the outputs defined above
    #q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="rgb_nn", maxSize=4, blocking=False)

    frame_depth = None
    bboxes = []

    while True:
        # sync inputs (since tryGet() is not used)
        in_nn = q_nn.get()
        in_depth = q_depth.get()
        
        if in_nn is not None:
            bboxes = in_nn.detections

        if in_depth is not None:
            frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
            frame_depth = np.ascontiguousarray(frame_depth)
            frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

        color = (255, 255, 255)

        if frame_depth is not None and in_nn is not None: # should be the case since .get() is used to sync inputs
            person_count = 0
            height = frame_depth.shape[0]
            width = frame_depth.shape[1]
            for bbox in bboxes:
                if bbox.label == 15:
                    person_count = person_count + 1
                    x1 = int(bbox.xmin * width)
                    x2 = int(bbox.xmax * width)
                    y1 = int(bbox.ymin * height)
                    y2 = int(bbox.ymax * height)
                    crop_frame = frame_depth[y1:y2, x1:x2]
                    #cv2.imshow("depth_crop", crop_frame)
                    print(f"Person {person_count} at {cv2.mean(crop_frame)[1]}") # BGR, green channel is depth (more or less)
            print(f"{person_count} persons")
            print()

        if cv2.waitKey(1) == ord('q'):
            break
