import numpy as np
import cv2
import depthai

pipeline = depthai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(500, 500)
cam.setInterleaved(False)
cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout_cam = pipeline.createXLinkOut()
xout_cam.setStreamName("cam")
cam.video.link(xout_cam.input)

device = depthai.Device(pipeline)
device.startPipeline()

q_cam = device.getOutputQueue("cam", maxSize=4, blocking=False)

while True:
    in_cam = q_cam.tryGet()
    frame = None

    if in_cam is not None:
        frame = in_cam.getCvFrame()

    if frame is not None:
        cv2.imshow("preview", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
