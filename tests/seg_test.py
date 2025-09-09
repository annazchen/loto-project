import cv2
import depthai as dai
import blobconverter
import time
import numpy as np

#grabbing yolov8 segmentation model
seg_blob = blobconverter.from_zoo(name = "yolov8s-seg", shaves = 10, data_type = "FP16")

#intializing pipeline
pipeline = dai.Pipeline()

#can specs
cam = pipeline.createColorCamera()
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setInterleaved(False)
cam.setFps(30)

#configure nn
cam.setPreviewSize(640, 640)

#segmentation nn
seg_nn = pipeline.createNeuralNetwork()
seg_nn.setBlobPath(seg_blob)
cam.preview.link(seg_nn.input)

#outputs
xout_seg = pipeline.createXLinkOut()
xout_seg.setStreamName("segmentation")
seg_nn.out.link(xout_seg.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize = 4, blocking = False)
    q_seg = device.getOutputQueue("segmentation", maxSize = 4, blocking = False)

    while True:
        in_rgb = q_rgb.tryGet()
        in_seg = q_seg.tryGet()

        if in_rgb is not None:
            raw_frame = in_rgb.getRaw()
            frame = np.array

        0, n = time.monotonic(), 0
        print("[INFO] running NN pipeline. Press q to quit")

        #convert seg output to mask
        masks = in_seg.getRaw()
