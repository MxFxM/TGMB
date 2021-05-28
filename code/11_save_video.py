import depthai as dai
import time

filename = f"{time.strftime('%Y%m%d%H%M%S')}.h265"

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xout = pipeline.createXLinkOut()

xout.setStreamName('h265')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
videoEnc.setDefaultProfilePreset((1920, 1080), 30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    # The .h265 file is a raw stream file (not playable yet)
    with open(filename, 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = q.get()  # Blocking call, will wait until a new data has arrived
                h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print(f"ffmpeg -framerate 30 -i {filename} -c copy filename.mp4")
