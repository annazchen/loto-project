#RUN TO CHECK SOCKET CONNECTIVITY AND ENSURE THAT FEED CAN BE VIEWED
#note this script is for DepthAI V3, V2
#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# define cam
cam_rgb = pipeline.create(dai.node.ColorCamera)

# cam properties
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)
cam_rgb.setVideoSize(1920, 1080)


# output queue straight from the node
videoQueue = cam_rgb.video.createOutputQueue()

#start the pipeline
pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()  # blocking
    cv2.imshow("video", videoIn.getCvFrame())
    if cv2.waitKey(1) == ord('q'): #press q to quit stream
        break





