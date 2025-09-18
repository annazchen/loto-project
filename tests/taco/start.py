import kagglehub

# Download latest version
path = kagglehub.dataset_download("vencerlanz09/taco-dataset-yolo-format")

print("Path to dataset files:", path)

#LOCATE FILE AND PLACE WITHIN WORKING DIRECTORY AFTER DOWNLOAD, OTHERWISE YAML CANNOT FIND IT
