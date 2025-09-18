import blobconverter

blob_path = blobconverter.from_onnx(
    model = "/home/user/Documents/loto-project/tests/taco/runs/detect/train3/weights/best.onnx",
    data_type = "FP16",
    shaves = 10
)


print(blob_path)
