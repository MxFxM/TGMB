#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - center camera
cam_center = pipeline.createColorCamera()
cam_center.setPreviewSize(300, 300)
cam_center.setBoardSocket(dai.CameraBoardSocket.RGB) # center camera on the oak-d
cam_center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_center.setInterleaved(False)
cam_center.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
cam_center.setVideoSize(300, 300)
cam_center.setSensorCrop(0.5, 0.5)

# Create outputs
xout_center = pipeline.createXLinkOut()
xout_center.setStreamName('center')
cam_center.preview.link(xout_center.input) # only output preview

xout_full = pipeline.createXLinkOut()
xout_full.setStreamName('full')
cam_center.video.link(xout_full.input) # output full video frame

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the grayscale frames from the outputs defined above
    q_center = device.getOutputQueue(name="center", maxSize=4, blocking=False)
    q_full = device.getOutputQueue(name="full", maxSize=4, blocking=False)

    frame_center = None
    frame_full = None

    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_center = q_center.tryGet()
        in_full = q_full.tryGet()

        if in_center is not None:
            frame_center = in_center.getCvFrame()
        if in_full is not None:
            frame_full = in_full.getCvFrame()

        # show the frames if available
        if frame_center is not None:
            cv2.imshow("center", frame_center)
        if frame_full is not None:
            cv2.imshow("full", frame_full)

        if cv2.waitKey(1) == ord('q'):
            break
