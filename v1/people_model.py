from ultralytics import YOLO

#loading yolov8n
model = YOLO("yolov8n.pt")

#train
model.train(data = "people.yaml", epochs = 100, imgsz = 640, batch = 16)
#debug
dataset = model.trainer.get_dataset()

#validate
stats = model.val()
print(stats)

#export model as openvino
model.export(format = "onnx", opset=11, simplify=True)