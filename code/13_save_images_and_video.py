import depthai as dai
import time

VIDEO_PATH = "./streams/"
video_filename = f"{VIDEO_PATH}{time.strftime('%Y%m%d%H%M%S')}.h265"

IMAGE_PATH = "./images/"

frame_counter = 0

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
imageEnc = pipeline.createVideoEncoder()
videoXout = pipeline.createXLinkOut()
imageXout = pipeline.createXLinkOut()

videoXout.setStreamName('h265')
imageXout.setStreamName('jpeg')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.H265_MAIN)
imageEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)

# Linking
camRgb.video.link(videoEnc.input)
camRgb.video.link(imageEnc.input)
videoEnc.bitstream.link(videoXout.input)
imageEnc.bitstream.link(imageXout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q_video = device.getOutputQueue(name="h265", maxSize=30, blocking=True)
    q_image = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    # The .h265 file is a raw stream file (not playable yet)
    with open(video_filename, 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = q_video.get()  # Blocking call, will wait until a new data has arrived
                h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file
                frame = q_image.get()
                frame_counter += 1
                if frame_counter == 30:
                    frame_counter = 0
                    with open(f"{IMAGE_PATH}{time.strftime('%Y%m%d%H%M%S')}.jpeg", "wb") as f:
                        f.write(bytearray(frame.getData()))
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print(f"ffmpeg -framerate 30 -i {video_filename} -c copy {video_filename[:-5]}.mp4")
