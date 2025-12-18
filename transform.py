import os
from PIL import Image
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (128, 128)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(path):
    img = Image.open(path).convert("L")
    return img

def resize_image(img):
    return img.resize(TARGET_SIZE)

def normalize_image(img_array):
    img_array = img_array.astype(np.float32)
    return img_array / 255.0

def process_image(input_path, output_path):
    img = load_image(input_path)
    img = resize_image(img)

    img_array = np.array(img)
    img_array = normalize_image(img_array)

    normalized_image = Image.fromarray((img_array * 255).astype(np.uint8), mode="L")
    normalized_image.save(output_path)

def process_dataset():
    ensure_dir(PROCESSED_DIR)

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, RAW_DIR)
                output_folder = os.path.join(PROCESSED_DIR, relative_path)
                ensure_dir(output_folder)

                output_file = os.path.splitext(file)[0] + ".png"
                output_path = os.path.join(output_folder, output_file)

                process_image(input_path, output_path)

if __name__ == "__main__":
    process_dataset()