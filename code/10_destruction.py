#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

from SALT import SALT
salt = SALT()

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

position_dict = {}
next_id = 1

laser_update_time = 0

video = None

if create_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25 # this is an assumed fps, works with mobilenet
    video = cv2.VideoWriter("video.avi",fourcc,fps,(networks[test_network]['size'],networks[test_network]['size']))

def getLowestNewestPosition(now):
    # this function is not done!
    # only look at unhealthy bees, so with varroa
    # if there are multiple newest ones, pick the one with the lowest id
    global position_dict
    position = None

    if len(position_dict) > 0:
        for entry in position_dict:
            if position_dict[entry]['time'] == now and position_dict[entry]['health'] == "healthy":
                position = position_dict[entry]['position']
                break
        return position
    else:
        return None
    return None


def checkPosition(position, now):
    global position_dict
    global next_id

    inthere = False
    thisid = 0

    for entry in position_dict:
        pos = position_dict[entry]['position']
        if position[0] > pos[0]-50 and position[0] < pos[0]+50 and position[1] > pos[1]-50 and position[1] < pos[1]+50:
            position_dict[entry]['position'] = position
            position_dict[entry]['time'] = now
            thisid = entry
            inthere = True
            break

    if not inthere:
        health_status = "unconfirmed"
        position_dict[next_id] = {'position':position,'time':now,'health':health_status}
        thisid = next_id
        next_id += 1

    return thisid

def updatePositions(now):
    global position_dict

    remove = []
    for entry in position_dict:
        if now - position_dict[entry]['time'] > 2: # not updated for 2 seconds
            remove.append(entry)

    for entry in remove:
        position_dict.pop(entry, None)

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
detection_nn.passthrough.link(xout_rgb.input)

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

    salt.set_laser('auto')

    while True:
        now = time.time()

        # use blocking get() call to catch frame and inference result synced
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()

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
                thisid = checkPosition(center, now)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"{texts[detection.label]} {thisid}", (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                in_var = q_var_out.get()
                result = np.array(in_var.getFirstLayerFp16())
                hey = health[np.argmax(result)]
                position_dict[thisid]['health'] = hey

                cv2.putText(frame, f"{hey}", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            if create_video:
                video.write(frame)
            cv2.imshow("rgb", frame)

            updatePositions(now)
           
            lowestPosition = getLowestNewestPosition(now)
            if lowestPosition is not None:
                #print(lowestPosition[0])
                x = 104 - (lowestPosition[0] / 300) * (104-59)
                y = (lowestPosition[1] / 300) * (82-42) + 42
                salt.set_angle('x', x)
                salt.set_angle('y', y)

        if cv2.waitKey(1) == ord('q'):
            if create_video:
                video.release()
            salt.close()
            break
