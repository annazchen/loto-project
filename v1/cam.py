import cv2
import depthai as dai
import time
import os 

#initializing pipeline
pipeline = dai.Pipeline()

#cam specs
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
cam.setIspScale(1, 3)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

#manual focus node
lens_ctrl = pipeline.create(dai.node.XLinkIn)
lens_ctrl.setStreamName('control')
lens_ctrl.out.link(cam.inputControl)

#output
#cam stream
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')
cam.preview.link(xout_rgb.input)

#preview
xout_preview = pipeline.createXLinkOut()
xout_preview.setStreamName("preview")
cam.preview.link(xout_preview.input)

#full res video
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam.video.link(xout_video.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue('rgb', maxSize = 4, blocking = False)
    q_preview = device.getOutputQueue("preview")
    q_video = device.getOutputQueue("video")

    ctrlQueue = device.getInputQueue('control')
    
    #set default focus = 120
    focus = 120 
    print(f"starting manual focus at {focus}")

    ctrl = dai.CameraControl()
    ctrl.setManualFocus(focus)
    ctrlQueue.send(ctrl)

    t0, n = time.monotonic(), 0
    print("[INFO] running NN pipeline. Press q to quit")

    while True:
        in_rgb = q_rgb.tryGet()
        in_preview = q_preview.tryGet()
        in_video = q_video.tryGet()

        #stream initializer, change in_[node type] as needed
        if in_video is not None:
            frame = in_video.getCvFrame()
            #bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('oak-1 max, rgb', frame)
        
        # c -> image capture
        # w -> zoom out
        # s -> zoom in
        cap = cv2.waitKey(1) & 0xFF
        if cap == ord('c'):
            filename = os.path.join(r"\Users\Anna.Chen\loto-2\data",f"person_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"saved {filename}")
        elif cap == ord('w'):
            focus = max(0, focus - 5)
            print(f"[INFO] focus -> {focus}")

            ctrl.setManualFocus(focus)
            ctrlQueue.send(ctrl)
        elif cap == ord('s'):
            focus = min(255, focus + 5)
            print(f"[INFO] focus -> {focus}")

            ctrl.setManualFocus(focus)
            ctrlQueue.send(ctrl)
        

cv2.destroyAllWindows()

