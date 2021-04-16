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

    for nextfile in tqdm(glob.glob("unlabeld/*.jpg")):
        name = nextfile[9:-4]
        #print(name)
        # load image into frame
        frame = cv2.imread(nextfile, cv2.IMREAD_COLOR)
        original_frame = frame.copy()
        # resize frame to 300x300
        frame = cv2.resize(frame, (300,300), interpolation=cv2.INTER_AREA)

        var_data = dai.NNData()
        var_data.setLayer("data", to_planar(frame, (300,300)))
        q_img_in.send(var_data)

        in_nn = q_nn.get()
        detections = in_nn.detections

        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder").text = "allimages"
        filename = ET.SubElement(annotation, "filename").text = f"{name}.jpg"
        path = ET.SubElement(annotation, "path").text = f"D:\\Hobby\\tgmb\\to-bee-or-not-to-bee\\allimages\\{name}.jpg"

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width").text = "640"
        height = ET.SubElement(size, "height").text = "480"
        depth = ET.SubElement(size, "depth").text = "3"

        segmented = ET.SubElement(annotation, "segmented").text = "0"

        for detection in detections:
            bbox = frame_norm(original_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            #det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            x1 = bbox[0]
            x2 = bbox[2]
            y1 = bbox[1]
            y2 = bbox[3]

            #print(x1, end=":")
            #print(y1)
            #print(x2, end=":")
            #print(y2)
            #print()

            bobject = ET.SubElement(annotation, "object")
            bname = ET.SubElement(bobject, "name").text = "bee"
            bpose = ET.SubElement(bobject, "pose").text = "Unspecified"
            btruncated = ET.SubElement(bobject, "truncated").text = "0"
            bdifficult = ET.SubElement(bobject, "difficult").text = "0"
            bndbox = ET.SubElement(bobject, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin").text = f"{x1}"
            ymin = ET.SubElement(bndbox, "ymin").text = f"{y1}"
            xmax = ET.SubElement(bndbox, "xmax").text = f"{x2}"
            ymax = ET.SubElement(bndbox, "ymax").text = f"{y2}"

        tree = ET.ElementTree(annotation)
        tree.write(f"labels/{name}.xml", pretty_print=True)

        os.remove(nextfile)
