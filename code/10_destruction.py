#!/usr/bin/env python3

# import different libraries
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

# this is the laser gimbal control module
from SALT import SALT
salt = SALT()

# select the networks to be used
test_network = 'mobilenet_1h'
test_health = 'fiveclass'

# store the captured frames in a video file
create_video = False

# a list of networks with working configurations
# one can be selected as the detection network and one for health classification
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

# parse command line arguments
# default paths to the networks with the selection from above
default_nn1 = str((Path(__file__).parent / Path(networks[test_network]['path'])).resolve().absolute())
default_nn2 = str((Path(__file__).parent / Path(networks[test_health]['path'])).resolve().absolute())
# create the parser
parser = argparse.ArgumentParser()
# options
parser.add_argument('mobilenet_path', nargs='?', help="Path to mobilenet detection network blob", default=default_nn1)
parser.add_argument('varroa_path', nargs='?', help="Path to varroa detection network blob", default=default_nn2)
parser.add_argument('-d', '--demo', action="store_true", help="Switch to demonstration mode, where the lowest tracked index is targeted", default=False)
args = parser.parse_args()

# tracking dictionary and tracking id
position_dict = {}
next_id = 1

# this would hold the video file if one is to be created
video = None

# the video file is initialized
if create_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25 # this is an assumed fps, works with mobilenet
    video = cv2.VideoWriter("video.avi",fourcc,fps,(networks[test_network]['size'],networks[test_network]['size']))

def getLowestNewestPosition(now):
    """
    This function returns the latest infested bee in the tracked positions.
    If the demo mode is selected, the lowest tracked index will be returned instead.
    """
    # get global variabels
    global position_dict
    position = None

    # if there is at least one tracked bee
    if len(position_dict) > 0:
        # iterate through the tracking list
        for entry in position_dict:
            if args.demo:
                # demo mode returns the first entry
                position = position_dict[entry]['position']
                break
            else:
                # find the latest timestamp and infested bee
                if position_dict[entry]['time'] == now and position_dict[entry]['health'] == "varroa":
                    # return its position
                    position = position_dict[entry]['position']
                    break
        return position
    else:
        return None
    return None


def checkPosition(position, now):
    """
    The main tracking function.
    New bees are added here and existing ones have their position updated.
    """
    # get global variables
    global position_dict
    global next_id

    inthere = False
    thisid = 0

    # for eack tracked bee
    for entry in position_dict:
        # get its previous position
        pos = position_dict[entry]['position']
        # if the new position is somewhat close
        if position[0] > pos[0]-50 and position[0] < pos[0]+50 and position[1] > pos[1]-50 and position[1] < pos[1]+50:
            # update the position and refresh the tracking time
            position_dict[entry]['position'] = position
            position_dict[entry]['time'] = now
            thisid = entry
            inthere = True
            break

    # if the bee does not exist yet, a new entry is created
    if not inthere:
        # health is unconfirmed to start with
        health_status = "unconfirmed"
        position_dict[next_id] = {'position':position,'time':now,'health':health_status}
        thisid = next_id
        next_id += 1

    # return the index of the bee
    return thisid

def updatePositions(now):
    """
    Each tracked bee has a time to live of 2 seconds.
    If the bee was not found for this amount of time, its entry is cleared from the dictionary.
    """
    # get global variables
    global position_dict
    remove = []

    # for every tracked bee
    for entry in position_dict:
        # if the entry was last updated more than 2 seconds ago
        if now - position_dict[entry]['time'] > 2: # not updated for 2 seconds
            # add that entry to the list of entries to be removed
            remove.append(entry)

    # remove all the marked entries
    for entry in remove:
        position_dict.pop(entry, None)

def getCenter(box):
    """
    This function calculates the center of a bounding box.
    """
    # get the corners of the box
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]

    # find the middle along the x axis
    center_x = int((x1+x2)/2)
    # find the middle along the y axis
    center_y = int((y1+y2)/2)

    # return that position
    return (center_x, center_y)

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

# Start defining a pipeline
pipeline = dai.Pipeline()

# define a source - color camera
cam_rgb = pipeline.createColorCamera()
# size is determined by the selected network
cam_rgb.setPreviewSize(networks[test_network]['size'], networks[test_network]['size'])
cam_rgb.setInterleaved(False)
# limit the frame rate
cam_rgb.setFps(40)

# define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createMobileNetDetectionNetwork()
# only high confidence bees are used
detection_nn.setConfidenceThreshold(0.7)
detection_nn.setBlobPath(args.mobilenet_path)
detection_nn.setNumInferenceThreads(2)
# the network shall run as fast as possible
detection_nn.input.setBlocking(False)
# input to the network is the preview stream of the camera with a defined size
cam_rgb.preview.link(detection_nn.input)

# the camera image is also pushed to the host
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
# the image is first passed through the network to have the input and output in sync
detection_nn.passthrough.link(xout_rgb.input)

# the results of the neural network are transfered to the host via an xlink
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# define a neural network to detect healthy or varroa on a cropped beed image
varroa_nn = pipeline.createNeuralNetwork()
varroa_nn.setBlobPath(args.varroa_path)

# the input to the classification network is coming from the host via an xlink
varroa_nn_xin = pipeline.createXLinkIn()
varroa_nn_xin.setStreamName("varroa_in")
varroa_nn_xin.out.link(varroa_nn.input)

# the results of the neural network are transfered to the host via an xlink
varroa_nn_xout = pipeline.createXLinkOut()
varroa_nn_xout.setStreamName("varroa_nn")
varroa_nn.out.link(varroa_nn_xout.input)

# labels for the classes of the detection and classification network
# the detection network also needs a "null" class
texts = ["null-0", "bee"]
# classification labels depend on the selected model from above
health = networks[test_health]['labels']

# once the pipeline is defined, the camera can be connected
with dai.Device(pipeline) as device:
    # start pipeline
    device.startPipeline()

    # output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    q_var_out = device.getOutputQueue(name="varroa_nn", maxSize=1, blocking=True)

    # the input queue is used to load the cropped images to the second stage network
    q_var_in = device.getInputQueue(name="varroa_in", maxSize=1, blocking=True)

    # to calculate the frames per second, a time is used
    start_time = time.monotonic()
    counter = 0

    detections = []
    frame = None

    def frame_norm(frame, bbox):
        """
        The function de-normalizes the bounding boxes from range 0 to 1 to the appropriate frame size.
        """
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    # the automatic laser mode is used
    # this means the laser will only turn on for a short pulse after the requested position was reached
    salt.set_laser('auto')

    # the main loop
    while True:
        # get the time once
        # every function relying on the time will be synced
        now = time.time()

        # use blocking get() call to catch frame and inference result synced
        in_rgb = q_rgb.get()
        in_nn = q_nn.get()

        # if there is a new frame from the camera
        if in_rgb is not None:
            # get the frame
            frame = in_rgb.getCvFrame()
            # print the framerate to it in the lower left corner
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - start_time)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

        # if there is detection data for the frame
        if in_nn is not None:
            # get the detections
            detections = in_nn.detections
            counter += 1

        # if the frame is available, draw bounding boxes on it and show the frame
        if frame is not None:
            # for every detected bee in the frame
            # (bees are the only possible detections with the models)
            for detection in detections:
                # normalize the bounding box
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                # get a cropped image of just the detection
                det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # pass the cropped image to the second stage nerural network for health classification
                var_data = dai.NNData()
                var_data.setLayer("0", to_planar(det_frame, (50, 50)))
                q_var_in.send(var_data)

                # since the network will take some time but we have to wait for it to syncronize the results
                # some other calculations are done in the meantime

                # calculate the center of the new detection
                center = getCenter(bbox)
                # draw a circle at the center (more like a dot)
                cv2.circle(frame, center, 2, (0, 255, 0), 2) # BGR
                # put the position into the tracking system
                thisid = checkPosition(center, now)

                # draw the bounding box on the frame
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # add the id that was assigned by the tracking system
                cv2.putText(frame, f"{texts[detection.label]} {thisid}", (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                # now get the results of the health classification
                in_var = q_var_out.get()
                result = np.array(in_var.getFirstLayerFp16())
                # find the most likely class
                hey = health[np.argmax(result)]
                # update the health status in the tracking system
                position_dict[thisid]['health'] = hey

                # print the health status on the frame
                cv2.putText(frame, f"{hey}", (bbox[0] + 10, bbox[1] + 40),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    
            # if a video is to be created
            if create_video:
                # add the new frame with all annotations
                video.write(frame)

            # show the frame on the host screen
            cv2.imshow("rgb", frame)

            # check if any tracked bee is no longer on the frame
            updatePositions(now)
           
            # get a position for the laser system
            lowestPosition = getLowestNewestPosition(now)
            # if a position was returned (at least one infested bee on the frame)
            if lowestPosition is not None:
                # calculate the angles for the gimbal
                # the limits have to be determined experimentally
                x = 104 - (lowestPosition[0] / 300) * (104-59)
                y = (lowestPosition[1] / 300) * (82-42) + 42
                # write the angles to SALT
                salt.set_angle('x', x)
                salt.set_angle('y', y)

        # if q was pressed
        if cv2.waitKey(1) == ord('q'):
            # stop the video if it is being created
            if create_video:
                video.release()
            # stop SALT
            salt.close()
            # end the main loop execution
            break
