#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

default_nn1 = str((Path(__file__).parent / Path('models/bee_detection_v2021.blob')).resolve().absolute())
default_nn2 = str((Path(__file__).parent / Path('models/varroa_v2021_202104111503.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('mobilenet_path', nargs='?', help="Path to mobilenet detection network blob", default=default_nn1)
parser.add_argument('varroa_path', nargs='?', help="Path to varroa detection network blob", default=default_nn2)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(40)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath(args.mobilenet_path)
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)
cam_rgb.preview.link(detection_nn.input)

# Define a neural network to detect healthy or varroa on a cropped beed image
varroa_nn = pipeline.createNeuralNetwork()
varroa_nn.setBlobPath(args.varroa_path)
varroa_nn_xin = pipeline.createXLinkIn() # input stream (push the cropped image to the nn)
varroa_nn_xin.setStreamName("varroa_in")
varroa_nn_xin.out.link(varroa_nn.input)
varroa_nn_xout = pipeline.createXLinkOut() # output stream (get results from the nn)
varroa_nn_xout.setStreamName("varroa_nn")
varroa_nn.out.link(varroa_nn_xout.input)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
if(args.sync):
    detection_nn.passthrough.link(xout_rgb.input)
else:
    cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# MobilenetSSD label texts
texts = ["null-0", "bee"]
health = ["healthy", "varroa"]

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_var_in = device.getInputQueue(name="varroa_in", maxSize=1, blocking=False)
    q_var_out = device.getOutputQueue(name="varroa_nn", maxSize=4, blocking=False)

    start_time = time.monotonic()
    counter = 0
    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


    while True:
        if(args.sync):
            # use blocking get() call to catch frame and inference result synced
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()
        else:
            # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()
            in_var = q_var_out.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - start_time)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if in_nn is not None:
            detections = in_nn.detections
            counter += 1

        if in_var is not None:
            result = np.array(in_var.getFirstLayerFp16())
            print(result, end='') # rename to dense_varroa later!
            hey = health[np.argmax(result)]
            print(hey)

        # if the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                var_data = dai.NNData()
                var_data.setLayer("0", to_planar(det_frame, (50, 50)))
                q_var_in.send(var_data)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, texts[detection.label], (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence*100)}%", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
