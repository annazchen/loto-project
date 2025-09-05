#RUN TO CHECK SOCKET CONNECTIVITY AND ENSURE THAT FEED CAN BE VIEWED


import depthai as dai
import cv2

#initialize pipeline
pipeline = dai.Pipeline()

#define cam specs

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(1920,1080) #1080p
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

xout =  pipeline.createXLinkOut()
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

#initializing stream
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name = "video", maxSize = 4, blocking = False)

    while True:
        in_frame = q.get() 
        frame = in_frame.getCvFrame()
        cv2.imshow("oak-1 max feed", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
cv2.destroyAllWindows()



