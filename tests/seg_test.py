import cv2
import depthai as dai
import blobconverter

#grabbing yolov8 segmentation model
seg_blob = blobconverter.from_zoo(name = "yolov8s-seg", shaves = 10)

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
