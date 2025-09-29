import depthai as dai
import cv2
import time
import numpy as np
import os
from blob_path import blob_path


num_detects = 0

#neural network input size
nn_size = 640

fps = 5

labels = ['person']
num_classes = len(labels)

#stores tuple (timestamp, frame)
frame_queue = []

#stores tuple (timestamp, detection)
det_queue = []

latest_detects = []
detections = []
max_q = 8

#default focus
focus = 120

#get length of detections list
def get_detections_count():
    global detections
    global num_detects
    num_detects = len(detections)
    return num_detects
    

#initialize pipeline
def cam_intialize(pipeline : dai.Pipeline, fps : int):
    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setIspScale(1, 3)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setInterleaved(False)
    cam.setFps(fps)
    return cam

#create neural network
def create_nn(pipeline : dai.Pipeline, cam : dai.node.ColorCamera, blob_path : str, nn_size : int):
    cam.setPreviewSize(nn_size, nn_size)
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    nn.input.setBlocking(False)
    nn.input.setQueueSize(1)
    return nn

#create manual focus node
def manual(pipeline : dai.Pipeline, cam : dai.node.ColorCamera):
    lens_ctrl = pipeline.create(dai.node.XLinkIn)
    lens_ctrl.setStreamName('control')
    lens_ctrl.out.link(cam.inputControl)
    return lens_ctrl

#output creator
def output(pipeline : dai.Pipeline, cam: dai.node.ColorCamera, stream_name : str):
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName(stream_name)
    if stream_name == "video":
        cam.video.link(xout.input)
    else:
        cam.preview.link(xout.input)
    return xout

#output: nn detections
def output_nn(pipeline : dai.Pipeline, cam : dai.node.ColorCamera, nn : dai.node.NeuralNetwork):
    xout_det = pipeline.create(dai.node.XLinkOut)
    xout_det.setStreamName("detections")

    cam.preview.link(nn.input)
    nn.out.link(xout_det.input)
    return xout_det

#output queue
def out_q(device : dai.Device, stream_name : str):
    return device.getOutputQueue(stream_name, maxSize = 4, blocking = False)

#input queue
def in_q(device : dai.Device, stream_name):
    return device.getInputQueue(stream_name)

#get frames
def get_frames(frames : dai.DataOutputQueue):
    return frames.tryGet()

#get number of detections
def get_detects(detects : list):
    return len(detects)

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

def main(stop_event = None):    
    global focus
    global latest_detects
    global detections

    #starting pipeline
    pipeline = dai.Pipeline()
    cam = cam_intialize(pipeline, fps)
    nn = create_nn(pipeline, cam, blob_path, nn_size)
    lens_ctrl = manual(pipeline, cam)

    #initializing outputs
    xout_rgb = output(pipeline, cam, "rgb")
    xout_det = output_nn(pipeline, cam, nn)
    xout_video = output(pipeline, cam, "video")

    with dai.Device(pipeline) as device:
        q_rgb = out_q(device, "rgb")
        q_det = out_q(device, "detections")
        q_video = out_q(device,"video")

        q_ctrl = in_q(device, "control")

        #starting manual focus control
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(focus)
        q_ctrl.send(ctrl)

        t0, n = time.monotonic(), 0
        print("[INFO] running NN pipeline. To quit all processes, press CTRL + c.")

        while not(stop_event and stop_event.is_set()):
            in_rgb = get_frames(q_rgb)
            in_det = get_frames(q_det)
            in_video = get_frames(q_video)


            #WIP, WILL BECOME A FUNCTION IN THE FUTURE
            if in_det is not None:
                #refresh detections
                detections.clear()

                #--- tensor config --- 
                tensor = in_det.getFirstLayerFp16() #tensor shape: [1 (batch dimension), 4 (for x,y,w,h) + # of classes , 8400]

                #debug: confirm tensor size assumptions are correct 
                #arr = np.array(in_det.getFirstLayerFp16())
                #print("[DEBUG] tensor length:", arr.shape)

                num_classes = len(labels)
                preds = np.array(tensor).reshape((4 + num_classes, -1)) #reshape tensor to [22, 8400]

                bbox = preds[:4, :]
                score = preds[4:, :]

                class_ids = np.argmax(score, axis = 0)
                confidences = np.max(score, axis = 0)

                bboxes = []
                confs = []
                classes = []

                for i in range(bbox.shape[1]):
                    conf = confidences[i]
                    if conf > 0.5:
                        cls = class_ids[i]
                        x, y, w, h = bbox[:, i]

                        xmin = int((x - w/2))
                        ymin = int((y - h/2))
                        xmax = int((x + w/2))
                        ymax = int((y + h/2))

                        bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        confs.append(float(conf))
                        classes.append(int(cls))

                indices = cv2.dnn.NMSBoxes(bboxes, confs, score_threshold = 0.5, nms_threshold = 0.4)

                for i in np.array(indices).flatten():
                    det = type('detection', (object,), {})()
                    det.xmin = bboxes[i][0] / nn_size
                    det.ymin = bboxes[i][1] / nn_size
                    det.xmax = (bboxes[i][0] + bboxes[i][2]) / nn_size
                    det.ymax = (bboxes[i][1] + bboxes[i][3]) / nn_size
                    det.confidence = confs[i]
                    det.label = classes[i]
                    detections.append(det)
                
                latest_detects = detections
                det_queue.append((in_det.getTimestamp(), detections))
                #debug: ensure detections are detecting
                #print(f"[DEBUG] got {len(detections)} detections")

                #for d in detections:
                    #print(f" - {labels[d.label]} ({d.confidence:.2f}) "
                    #    f"at [{d.xmin:.2f}, {d.ymin:.2f}, {d.xmax:.2f}, {d.ymax:.2f}]")
                #remove oldest detection frame    
                if len(det_queue) > max_q:
                    det_queue.pop(0)

            if in_video is not None:
                #getting timestamp and frame
                frame_queue.append((in_video.getTimestamp(), in_video.getCvFrame()))
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
                    #if cannot find, use last known detection
                    detections_to_draw = latest_detects

                if detections:
                    d = detections[0]
                    #print(f"[DEBUG] number of detections: {len(detections)}")
                    #print(f"[DEBUG] box: {d.xmin:.2f}, {d.ymin:.2f}, {d.xmax:.2f}, {d.ymax:.2f}")

                draw(frame, detections_to_draw)
                cv2.imshow("OAK-1 Max - YOLOv8n", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                focus = max(0, focus - 5)
                print(f"[INFO] focus -> {focus}")
                ctrl.setManualFocus(focus)
                q_ctrl.send(ctrl)
            elif key == ord('s'):
                focus = min(255, focus + 5)
                print(f"[INFO] focus -> {focus}")
                ctrl.setManualFocus(focus)
                q_ctrl.send(ctrl)
            #press c to take snapshot
            elif key == ord('c'):
                filename = os.path.join(r"\Users\Anna.Chen\loto-2\data",f"person_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"saved {filename}")

    cv2.destroyAllWindows()
    time.sleep(0.1)
    print("exiting people_detect...")    
if __name__ == "__main__":
    main()
    
    

    