import time 
import cv2
import depthai as dai
import blobconverter


#grabbing mobilenetssd
mn = blobconverter.from_zoo(name = "mobilenet-ssd", shaves = 10)


#start pipeline
pipeline = dai.Pipeline()

#cam specs
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

#configure nn
nn_size = 300
cam.setPreviewSize(nn_size, nn_size)

nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(mn)
nn.setConfidenceThreshold(0.5)

#runtime

labels = [
"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
"bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
"horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
"train", "tvmonitor"
]

def draw(frame, detections):
    for det in detections:
        h,w = frame.shape[:2]
        x1, y1, x2, y2 = int(det.xmin * w), int(det.ymin * h), int(det.xmax * w), int(det.ymax * h)
        lbl = det.label if 0 <= det.label < len(labels) else det.label
        conf = int(det.confidence * 100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,f"{labels[lbl] if isinstance(lbl, int) else lbl} {conf}%", (x1 + 5, max(y1-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #cv2.rectangle(frame, f"{labels[lbl] if isinstance(lbl, int) else lbl} {conf}%",
                      #(x1 + 5, max(y1 -6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,
                      #cv2.LINE_AA)


#outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_det = pipeline.createXLinkOut()
xout_det.setStreamName("detections")

cam.preview.link(nn.input)
nn.out.link(xout_det.input)



"""with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking = False)

    t0, n = time.monotonic(), 0
    print("[INFO] running NN pipeline. Press q to quit")

    while True:
        in_rgb = q_rgb.tryGet()
        in_det = q_det.tryGet()

        frame = in_rgb.getCvFrame() if in_rgb is not None else None
        if in_det is not None:
            dets = in_det.detections

            #debug
            for det in dets:
                print(f"Detecting: {labels[det.label]} ({det.confidence: .2f} )")
            n += 1 
        else:
            dets = []
        if frame is not None:
            draw(frame, dets)
            dt = time.monotonic() - t0
            if dt > 0:
                cv2.putText(frame, f"NN fps: {n/dt:.2f}", (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("oak-1 max - mobilenet-ssd", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""

frame_queue = []
det_queue = []

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking=False)

    t0, n = time.monotonic(), 0
    print("[INFO] running NN pipeline. Press q to quit")

    while True:
        in_rgb = q_rgb.tryGet()
        in_det = q_det.tryGet()

        #debug
        if in_det is not None:
            dets = in_det.detections
            for det in dets:
                print(f"Detecting: {labels[det.label]} ({det.confidence: .2f} )")

        if in_rgb is not None:
            frame_queue.append((in_rgb.getTimestamp(), in_rgb.getCvFrame()))

        if in_det is not None:
            det_queue.append((in_det.getTimestamp(), in_det.detections))

        # match latest detection to closest frame
        if frame_queue and det_queue:
            f_ts, frame = frame_queue.pop(0)
            # find detection closest in time
            closest_det = min(det_queue, key=lambda x: abs(x[0]-f_ts))
            detections = closest_det[1]

            
            draw(frame, detections)
            cv2.imshow("OAK-1 Max - MobileNetSSD", frame)

        if cv2.waitKey(1) == ord('q'):
            break


cv2.destroyAllWindows()
