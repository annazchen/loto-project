from ultralytics import YOLO

#loading yolov8n
model = YOLO("yolov8n.pt")

#train
model.train(data = "asl.yaml", epochs = 50, imgsz = 640, batch = 5)
#debug
dataset = model.trainer.get_dataset()

#validate
stats = model.val()
print(stats)

#export model as openvino
model.export(format = "openvino", opset=11, simplify=True)