import depthai as dai
import time

IMAGE_PATH = "./images/"

frame_counter = 0

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)

imageEnc = pipeline.createVideoEncoder()
imageEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(imageEnc.input)

imageXout = pipeline.createXLinkOut()
imageXout.setStreamName("jpeg")
imageEnc.bitstream.link(imageXout.input)

with dai.Device(pipeline) as device:
    q_image = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    while True:
        frame = q_image.get()

        frame_counter += 1
        if frame_counter == 30:
            frame_counter = 0
            with open(f"{IMAGE_PATH}{time.strftime('%Y%m%d%H%M%S')}", "wb") as f:
                f.write(bytearray(frame.getData()))

