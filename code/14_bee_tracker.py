#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

default_nn = str((Path(__file__).parent / Path('models/frozen_inference_graph.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('mobilenet_path', nargs='?', help="Path to mobilenet detection network blob", default=default_nn)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=True)
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

# Define an object tracker
beeTracker = pipeline.createObjectTracker()
beeTracker.setDetectionLabelsToTrack([1])  # track only bees
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
beeTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# possible options: SMALLEST_ID, UNIQUE_ID
beeTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)
# link to image and detections
detection_nn.passthrough.link(beeTracker.inputDetectionFrame)
detection_nn.out.link(beeTracker.inputDetections)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

trackerOut = pipeline.createXLinkOut()
trackerOut.setStreamName("tracklets")
beeTracker.out.link(trackerOut.input)

# MobilenetSSD label texts
texts = ["null-0", "bee"]

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
	q_tracker = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)

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
		in_tracklets = q_tracker.get()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - start_time)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if in_nn is not None:
            detections = in_nn.detections
            counter += 1
		
		if in_tracklets is not None:
			trackletsData = in_tracklets.tracklets

        # if the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, texts[detection.label], (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence*100)}%", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.imshow("rgb", frame)
		
			for t in trackletsData:
				roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
				x1 = int(roi.topLeft().x)
				y1 = int(roi.topLeft().y)
				x2 = int(roi.bottomRight().x)
				y2 = int(roi.bottomRight().y)

				try:
					label = labelMap[t.label]
				except:
					label = t.label

				cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

			cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

			cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break
