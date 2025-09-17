from ultralytics import YOLO

#loading yolov8n
model = YOLO("yolov8n.pt")

#(person)al training
model.train(data = "coco8.yaml", epochs = 50, imgsz = 640, batch = 5, classes = [0])

#validation
stats = model.val()
print(stats)

#export model
model.export(format = "openvino", opset=11, simplify=True)
