import blobconverter

blob_path = blobconverter.from_onnx(xml = "/home/user/Documents/loto-project/runs/detect/train15/weights/best_openvino_model/best.xml",
                          bin = "/home/user/Documents/loto-project/runs/detect/train15/weights/best_openvino_model/best.bin",
                          data_type="FP16",
                          shaves=10)

print()