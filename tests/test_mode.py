import time 
import cv2
import depthai as dai
import blobconverter


#grabbing mobilenetssd
mn = blobconverter.from_zoo(name = "mobilenetssd", shaves = 10)


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

#outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_det = pipeline.createXLinkOut()
xout_det.setStreamName("detections")
nn.out.link(xout_det.input)

#runtime

labels = [
"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
"bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
"horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
"train", "tvmonitor"
]

def draw(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
        lbl = det.label if 0 <= det.label < len(labels) else det.label
        conf = int(det.confidence * 100)

        cv2.rectangle(frame, f"{labels[lbl] if isinstance(lbl, int) else lbl} {conf}%",
                      (x1 + 5, max(y1 -6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,
                      cv2.LINE_AA)

with dai.Device(pipeline) as device:
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
            n += 1 
        else:
            dets = []
        if frame is not None:
            draw(frame, dets)
            dt = time.monotonic() - t0
            if dt > 0:
                cv2.putText(frame, f"NN fps: {n/dt:.2f}", (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1. cv2.LINE_AA)
            cv2.imshow("oak-1 max - mobilenet-ssd", frame)

        if cv2.wait(1) & 0xFF == ord('q'):
            break
        
