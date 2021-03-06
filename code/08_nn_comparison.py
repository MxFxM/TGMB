#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

test_network = 'mobilenet_1h'
test_health = 'fiveclass'
create_video = False

networks = {
        'mobilenet_1h':{'path':'models/bee_detection_v2021_202104141915.blob','size':300},
        'mobilenet_4h':{'path':'models/bee_detection_v2021_202104142335.blob','size':300},
        'resnet_4s':{'path':'models/bee_detection_v2021_202104152318_resnet_4s.blob','size':640},
        'resnet_6s':{'path':'models/bee_detection_v2021_202104152318_resnet_6s.blob','size':640},
        'resnet_8s':{'path':'models/bee_detection_v2021_202104152318_resnet_8s.blob','size':640},
        'resnet_10s':{'path':'models/bee_detection_v2021_202104152318_resnet_10s.blob','size':640},
        'twoclass_old':{'path':'models/varroa_v2021_202104111503.blob','labels':["healthy", "varroa"]},
        'fiveclass':{'path':'models/varroa_v2021_202104112109_5classes.blob','labels':["ants","healthy","noqueen","robbed","varroa"]}
}

position_list = []
position_update = []
position_ids = []
next_id = 1

video = None

if create_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25 # this is an assumed fps, works with mobilenet
    video = cv2.VideoWriter("video.avi",fourcc,fps,(networks[test_network]['size'],networks[test_network]['size']))

def checkPosition(position):
    global position_list
    global position_update
    global position_ids
    global next_id

    x = position[0]
    y = position[1]
    
    inthere = False
    now = time.time()
    thisid = 0

    for n, pos in enumerate(position_list):
        if x > pos[0]-30 and x < pos[0]+30 and y > pos[1]-30 and y < pos[1]+30:
            position_list[n] = position
            position_update[n] = now
            thisid = position_ids[n]
            inthere = True
            break

    if not inthere:
        position_list.append(position)
        position_update.append(now)
        position_ids.append(next_id)
        thisid = next_id
        next_id += 1

    return thisid

def updatePositions():
    global position_list
    global position_update
    global position_ids

    if len(position_update) > 0:
        now = time.time()
        remove = []

        for n, last_time in enumerate(position_update):
            if now - last_time > 3: # not updated for 3 seconds
                remove.append(n) # add index to remove list

        position_list = [pos for n, pos in enumerate(position_list) if n not in remove]
        position_update = [tim for n, tim in enumerate(position_update) if n not in remove]
        position_ids = [pid for n, pid in enumerate(position_ids) if n not in remove]

def getCenter(box):
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)
    return (center_x, center_y)

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

default_nn1 = str((Path(__file__).parent / Path(networks[test_network]['path'])).resolve().absolute())
default_nn2 = str((Path(__file__).parent / Path(networks[test_health]['path'])).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('mobilenet_path', nargs='?', help="Path to mobilenet detection network blob", default=default_nn1)
parser.add_argument('varroa_path', nargs='?', help="Path to varroa detection network blob", default=default_nn2)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(networks[test_network]['size'], networks[test_network]['size'])
cam_rgb.setInterleaved(False)
cam_rgb.setFps(40)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setConfidenceThreshold(0.7)
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
health = networks[test_health]['labels']

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_var_in = device.getInputQueue(name="varroa_in", maxSize=1, blocking=True)
    q_var_out = device.getOutputQueue(name="varroa_nn", maxSize=1, blocking=True)

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

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - start_time)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        if in_nn is not None:
            detections = in_nn.detections
            counter += 1

        # if the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            for detection in detections:
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                var_data = dai.NNData()
                var_data.setLayer("0", to_planar(det_frame, (50, 50)))
                q_var_in.send(var_data)

                center = getCenter(bbox)
                cv2.circle(frame, center, 2, (0, 255, 0), 2) # BGR
                thisid = checkPosition(center)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"{texts[detection.label]} {thisid}", (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                #cv2.putText(frame, f"{int(detection.confidence*100)}%", (bbox[0] + 10, bbox[1] + 40),
                #            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                in_var = q_var_out.get()
                result = np.array(in_var.getFirstLayerFp16())
                hey = health[np.argmax(result)]

                cv2.putText(frame, f"{hey}", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            if create_video:
                video.write(frame)
            cv2.imshow("rgb", frame)
            updatePositions()
            #print(position_list)

        if cv2.waitKey(1) == ord('q'):
            if create_video:
                video.release()
            break
