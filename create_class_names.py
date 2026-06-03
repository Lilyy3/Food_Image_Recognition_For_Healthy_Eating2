import os
from pathlib import Path
import kagglehub

DATASET_PATH = kagglehub.dataset_download("kmader/food41")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")

class_names = sorted(os.listdir(IMAGE_DIR))

output_path = Path("class_names.txt")

with open(output_path, "w", encoding="utf-8") as file:
    for class_name in class_names:
        file.write(class_name + "\n")

print("Saved class_names.txt")
print("Number of classes:", len(class_names))
