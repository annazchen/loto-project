import blobconverter

blob_path = blobconverter.from_onnx(
    model = r'\Users\Anna.Chen\github\loto-project\v1\runs\detect\train3\weights\best.onnx',
    data_type = "FP16",
    shaves = 10
)


print(blob_path)
