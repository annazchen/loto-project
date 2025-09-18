import depthai as dai
import cv2
import time
import numpy as np

blob_path = "/home/user/.cache/blobconverter/best_openvino_2022.1_10shave.blob"

#start pipeline
pipeline = dai.Pipeline()

#cam specs
cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

#config nn
nn_size = 640
cam.setPreviewSize(640, 640)

nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(blob_path)
nn.input.setBlocking(False)
nn.input.setQueueSize(1)

#outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_det = pipeline.create(dai.node.XLinkOut)
xout_det.setStreamName("detections")

cam.preview.link(nn.input)
nn.out.link(xout_det.input)

labels = ['Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton', 
          'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 'Plastic bag - wrapper',
          'Plastic container', 'Pop tab', 'Straw', 'Styrofoam piece', 'Unlabeled litter']


#draw bounding boxes
def draw(frame, detections):
    for det in detections:
        h,w = frame.shape[:2]
        x1, y1, x2, y2 = int(det.xmin * w), int(det.ymin * h), int(det.xmax * w), int(det.ymax * h)
        if 0 <= det.label < len(labels):
            lbl = labels[det.label]
        else:
            lbl = f"cls_{det.label}"
        conf = int(det.confidence * 100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,f"{labels[lbl] if isinstance(lbl, int) else lbl} {conf}%", (x1 + 5, max(y1-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

frame_queue = []
det_queue = []
latest_detects = []
detections = []
max_q = 8

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking=False)

    t0, n = time.monotonic(), 0
    print("[INFO] running NN pipeline. Press q to quit")

    while True:
        in_rgb = q_rgb.tryGet()
        in_det = q_det.tryGet()

        if in_det is not None:
            detections = []
            tensor = in_det.getFirstLayerFp16() #tensor shape: [1 (batch dimension), 4 (for x,y,w,h) + # of classes , 8400]

            #debug: confirm tensor size assumptions are correct 
            #arr = np.array(in_det.getFirstLayerFp16())
            #print("[DEBUG] tensor length:", arr.shape)

            num_classes = len(labels)
            preds = np.array(in_det.getFirstLayerFp16()).reshape((4 + num_classes, -1)) #reshape tensor to [22, 8400]

            bbox = preds[:4, :]
            score = preds[4:, :]

            class_ids = np.argmax(score, axis = 0)
            confidences = np.max(score, axis = 0)

            for i in range(bbox.shape[1]):
                conf = confidences[i]
                if conf > 0.5:
                    cls = class_ids[i]
                    x, y, w, h = bbox[:, i]

                    det = type('Detection', (object,), {})()
                    det.xmin = (x - w/2) / nn_size
                    det.ymin = (y - h/2) / nn_size 
                    det.xmax = (x + w/2) / nn_size
                    det.ymax = (y + h/2) / nn_size
                    det.confidence = float(conf)
                    det.label = int(cls)
                    detections.append(det)
            latest_detects = detections
            det_queue.append((in_det.getTimestamp(), detections))
            
            #debug: ensure detections are detecting
            print(f"[DEBUG] got {len(detections)} detections")

        

            for d in detections:
                print(f" - {labels[d.label]} ({d.confidence:.2f}) "
                    f"at [{d.xmin:.2f}, {d.ymin:.2f}, {d.xmax:.2f}, {d.ymax:.2f}]")
            if len(det_queue) > max_q:
                det_queue.pop(0)

        if in_rgb is not None:
            #getting frame timestamp and frame
            frame_queue.append((in_rgb.getTimestamp(), in_rgb.getCvFrame()))
            if len(frame_queue) > max_q:
                frame_queue.pop(0)
        
        #match latest detection to closest frame
        if frame_queue:
            f_ts, frame = frame_queue.pop(0)

            if det_queue:
                # find closest detection timestamp
                closest_det = min(det_queue, key=lambda x: abs(x[0] - f_ts))
                detections_to_draw = closest_det[1]
            else:
                # fallback: use last known detections
                detections_to_draw = latest_detects

            if detections:
                d = detections[0]
                print(f"[DEBUG] box: {d.xmin:.2f}, {d.ymin:.2f}, {d.xmax:.2f}, {d.ymax:.2f}")


            draw(frame, detections_to_draw)
            cv2.imshow("OAK-1 Max - YOLOv8n", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
