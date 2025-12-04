"""
Organize images by label and retrain the model
"""
import os
import shutil
from pathlib import Path

# Mapping filename to label (Romanian sign names)
LABEL_MAPPING = {
    'stop.png': 'STOP',
    'viteza30.png': 'VITEZA_30',
    'trecerepietoni.png': 'TRECERE_PIETONI',
    'scoala.png': 'SCOALA',
    'sensgiratoriu.png': 'SENS_GIRATORIU',
    'vireazadreapta.png': 'VIRAJA_DREAPTA',
    'ocoliredreapta.png': 'OCOLI_DREAPTA',
    'faraprorietate.png': 'FARA_PRIORITATE',
    'priorietate.png': 'PRIORITATE',
    'cedeaza.png': 'CEDEAZA',
    'curbastanga.png': 'CURBA_STANGA',
    'atentiedreapta.png': 'ATENTIE_DREAPTA',
    'oameni.png': 'OAMENI',
    'ambelesensuri.png': 'AMBELE_SENSURI',
}

RAW_FOLDER = 'data/raw'
LABELED_FOLDER = 'data/labeled'

# Create label subfolders and copy images
for filename, label in LABEL_MAPPING.items():
    label_dir = os.path.join(LABELED_FOLDER, label)
    os.makedirs(label_dir, exist_ok=True)
    
    src = os.path.join(RAW_FOLDER, filename)
    dst = os.path.join(label_dir, filename)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[OK] {filename} -> {label}/")
    else:
        print(f"[SKIP] {filename} not found")

print(f"\n[INFO] Images organized in {LABELED_FOLDER}/")

# Update labels.txt with correct mapping
labels_list = sorted(set(LABEL_MAPPING.values()))
with open('data/labels.txt', 'w', encoding='utf-8') as f:
    for label in labels_list:
        f.write(label + '\n')

print(f"[INFO] Updated data/labels.txt with {len(labels_list)} labels:")
for i, label in enumerate(labels_list):
    print(f"  {i}: {label}")

# Now retrain the model
print("\n[INFO] Retraining model...")

from src.neural_network.train import train_model

train_model(labeled_data_dir=LABELED_FOLDER)

print("[OK] Model retraining complete!")
