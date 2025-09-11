import cv2
import depthai as dai
import time
import os 

#initializing pipeline
pipeline = dai.Pipeline()

#cam specs
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

#output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')
cam.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue('rgb', maxSize = 4, blocking = False)

    t0, n = time.monotonic(), 0
    print("[INFO] running NN pipeline. Press q to quit")

    while True:
        in_rgb = q_rgb.tryGet()

        #stream initializer
        if in_rgb is not None:
            rgb = in_rgb.getCvFrame()
            bw = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            cv2.imshow('oak-1 max, b/w', bw)
        
        #regular image capture
        cap = cv2.waitKey(1) & 0xFF
        if cap == ord('c'):
            filename = os.path.join("/home/user/Documents/loto-project/v1/dataset/2_lock",f"2_{int(time.time())}.jpg")
            cv2.imwrite(filename, bw)
            print(f"saved {filename}")

        #timed image capture
        if cap == ord('d'):
            print("starting timed capture:")
            t = 3
            while (t):
                print(t)
                time.sleep(1)
                t -= 1
            filename = os.path.join("/home/user/Documents/loto-project/v1/dataset/2_lock",f"2_{int(time.time())}.jpg")
            cv2.imwrite(filename, bw)
            print(f"saved {filename}")
        
        #multishot
        if (cap == ord('f')):
            multi = True
        elif (cap == 255):
            multi = False
        if multi:
            filename = os.path.join("/home/user/Documents/loto-project/v1/dataset/2_lock",f"2_{int(time.time())}.jpg")
            cv2.imwrite(filename, bw)
            print(f"saved {filename}")
            time.sleep(0.5)

        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

