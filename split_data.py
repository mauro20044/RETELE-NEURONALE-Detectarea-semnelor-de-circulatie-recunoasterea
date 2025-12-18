import os
import shutil
import random
from pathlib import Path

# Configurare directoare
RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
VALIDATION_DIR = "data/validation"

# Proporții pentru split
TRAIN_RATIO = 0.7
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.1

def ensure_dir(path):
    """Creează directorul dacă nu există"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_image_files(directory):
    """Obține lista de imagini din director"""
    image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def split_and_copy_images():
    """Distribuie imagini în train, test și validation"""
    
    # Creează directoarele de destinație
    ensure_dir(TRAIN_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(VALIDATION_DIR)
    
    # Obține lista de imagini din folderul raw
    image_files = get_image_files(RAW_DIR)
    
    if not image_files:
        print("Nu s-au găsit imagini în folderul raw!")
        return
    
    print(f"Total imagini găsite: {len(image_files)}")
    
    # Amestecă lista de imagini
    random.shuffle(image_files)
    
    # Calculează indicii pentru split
    train_count = int(len(image_files) * TRAIN_RATIO)
    test_count = int(len(image_files) * TEST_RATIO)
    
    train_files = image_files[:train_count]
    test_files = image_files[train_count:train_count + test_count]
    validation_files = image_files[train_count + test_count:]
    
    # Funcție pentru copiere fișiere
    def copy_files(file_list, destination, folder_name):
        for src_file in file_list:
            try:
                filename = os.path.basename(src_file)
                dst_file = os.path.join(destination, filename)
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Eroare la copiere {src_file}: {e}")
        print(f"{folder_name}: {len(file_list)} imagini copiate")
    
    # Copiază fișierele în folderele respective
    copy_files(train_files, TRAIN_DIR, "Train")
    copy_files(test_files, TEST_DIR, "Test")
    copy_files(validation_files, VALIDATION_DIR, "Validation")
    
    print("\nDistribuire completă!")
    print(f"Train: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)")
    print(f"Test: {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)")
    print(f"Validation: {len(validation_files)} ({len(validation_files)/len(image_files)*100:.1f}%)")

if __name__ == "__main__":
    split_and_copy_images()
