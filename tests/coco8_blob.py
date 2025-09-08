from blobconverter import from_openvino

blob_path = from_openvino(xml = "/home/user/Documents/loto-project/runs/detect/train14/weights/best_openvino_model/best.xml",
                          bin = "/home/user/Documents/loto-project/runs/detect/train14/weights/best_openvino_model/best.bin",
                          data_type="FP16",
                          shaves=6)

print()