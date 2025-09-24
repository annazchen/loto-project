import blobconverter

blob_path = blobconverter.from_onnx(
    model = '/home/user/Documents/loto-project/v1/runs/detect/train/weights/best.onnx' ,
    data_type = "FP16",
    shaves = 10
)


print(blob_path)
