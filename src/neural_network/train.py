"""
Script principal pentru antrenarea rețelei neuronale
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.neural_network.model import TrafficSignCNN
from src.preprocessing.image_preprocessing import load_dataset, ImagePreprocessor

try:
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    def to_categorical(y, num_classes):
        """Fallback pentru to_categorical"""
        result = np.zeros((len(y), num_classes))
        for i, label in enumerate(y):
            result[i, int(label)] = 1
        return result


def plot_training_history(history, save_path="training_history.png"):
    """
    Plotează istoricul antrenării
    
    Args:
        history: Istoricul returnat de model.fit()
        save_path: Calea pentru salvare
    """
    if not hasattr(history, 'history'):
        print("Fara date de istoric pentru plotare")
        return
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Graficul salvat: {save_path}")
        plt.close()
    except Exception as e:
        print(f"Eroare la plotare: {e}")


def _train_from_labeled(labeled_data_dir):
    """Train from labeled directory structure (subfolder = label)"""
    
    print(f"\n[INFO] Loading images from {labeled_data_dir}/...")
    
    # Get list of labels (subdirectories)
    labels_list = sorted([d for d in os.listdir(labeled_data_dir) 
                         if os.path.isdir(os.path.join(labeled_data_dir, d))])
    num_classes = len(labels_list)
    
    print(f"[INFO] Found {num_classes} classes: {labels_list}")
    
    # Load images and labels
    x_all = []
    y_all = []
    preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
    
    for label_idx, label in enumerate(labels_list):
        label_dir = os.path.join(labeled_data_dir, label)
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(label_dir, filename)
                try:
                    img = preprocessor.preprocess(img_path)
                    x_all.append(img)
                    y_all.append(label_idx)
                    print(f"  [OK] {label}/{filename}")
                except Exception as e:
                    print(f"  [ERROR] {label}/{filename}: {e}")
    
    if len(x_all) == 0:
        print("[ERROR] No images loaded!")
        return
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    print(f"\n[INFO] Loaded {len(x_all)} images")
    
    # Split into train (70%), val (15%), test (15%)
    np.random.seed(42)
    indices = np.random.permutation(len(x_all))
    
    train_split = int(0.7 * len(x_all))
    val_split = int(0.85 * len(x_all))
    
    x_train = x_all[indices[:train_split]]
    y_train = y_all[indices[:train_split]]
    
    x_val = x_all[indices[train_split:val_split]]
    y_val = y_all[indices[train_split:val_split]]
    
    x_test = x_all[indices[val_split:]]
    y_test = y_all[indices[val_split:]]
    
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    # Reshape for model
    x_train = x_train.reshape(-1, 128, 128, 1)
    x_val = x_val.reshape(-1, 128, 128, 1)
    x_test = x_test.reshape(-1, 128, 128, 1)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Build and train model
    print("\n[INFO] Building model...")
    cnn = TrafficSignCNN(input_shape=(128, 128, 1), num_classes=num_classes)
    cnn.build_model()
    cnn.compile_model()
    
    print("\n[INFO] Training...")
    history = cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=50,
        batch_size=8
    )
    
    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = cnn.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    print("\n[INFO] Saving model...")
    model_path = "models/traffic_sign_model.h5"
    os.makedirs("models", exist_ok=True)
    cnn.save_model(model_path)
    
    # Save labels
    with open('data/labels.txt', 'w', encoding='utf-8') as f:
        for label in labels_list:
            f.write(label + '\n')
    
    plot_training_history(history, "training_history.png")
    
    print("\n" + "=" * 60)
    print("ANTRENAMENT COMPLETAT!")
    print("=" * 60)


def train_model(labeled_data_dir=None):
    """Main training function"""
    
    print("=" * 60)
    print("ANTRENAMENT RETEA NEURALA - SEMNE DE CIRCULATIE")
    print("=" * 60)
    
    if labeled_data_dir and os.path.exists(labeled_data_dir):
        # Load from labeled directory structure
        return _train_from_labeled(labeled_data_dir)
    
    # Directoare standard
    train_dir = "data/train"
    val_dir = "data/validation"
    test_dir = "data/test"
    
    # Verifică dacă directoarele există
    if not os.path.exists(train_dir):
        print(f"Eroare: Directorul {train_dir} nu exista!")
        return
    
    print("\n[INFO] Incarcarea datelor...")
    
    # Incarca datele
    x_train, train_files = load_dataset(train_dir)
    print(f"   [INFO] Train: {len(x_train)} imagini")
    
    x_val, val_files = load_dataset(val_dir) if os.path.exists(val_dir) else (None, [])
    if x_val is not None:
        print(f"   [INFO] Validation: {len(x_val)} imagini")
    else:
        print(f"   Validation: Director nu exista, folosesc split din train")
        split_idx = int(len(x_train) * 0.8)
        x_val = x_train[split_idx:]
        x_train = x_train[:split_idx]
    
    x_test, test_files = load_dataset(test_dir) if os.path.exists(test_dir) else (None, [])
    if x_test is not None:
        print(f"   Test: {len(x_test)} imagini")
    else:
        print(f"   Test: Director nu exista")
    
    # Reshape pentru CNN
    x_train = x_train.reshape(-1, 128, 128, 1)
    x_val = x_val.reshape(-1, 128, 128, 1) if x_val is not None else None
    if x_test is not None:
        x_test = x_test.reshape(-1, 128, 128, 1)
    
    # Creaza etichete dummy (toate clasa 0) - pentru compatibilitate
    print("\n[INFO] Crearea etichetelor...")
    num_classes = 10
    y_train = to_categorical(np.zeros(len(x_train), dtype=int), num_classes)
    y_val = to_categorical(np.zeros(len(x_val), dtype=int), num_classes) if x_val is not None else None
    y_test = to_categorical(np.zeros(len(x_test), dtype=int), num_classes) if x_test is not None else None
    
    # Construieste modelul
    print("\n[INFO] Constructie model...")
    cnn = TrafficSignCNN(input_shape=(128, 128, 1), num_classes=num_classes)
    cnn.build_model()
    cnn.compile_model()
    
    print("\n[INFO] Rezumatul modelului:")
    summary = cnn.get_model_summary()
    if summary:
        print(summary)
    
    # Antrenare
    print("\n[INFO] Antrenament in curs...")
    history = cnn.train(
        x_train, y_train,
        x_val, y_val,
        epochs=50,
        batch_size=8
    )
    
    # Evaluare
    if x_test is not None:
        print("\n[INFO] Evaluare pe test set...")
        test_loss, test_acc = cnn.evaluate(x_test, y_test)
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
    
    # Salveaza modelul
    print("\n[INFO] Salvare model...")
    model_path = "models/traffic_sign_model.h5"
    os.makedirs("models", exist_ok=True)
    cnn.save_model(model_path)
    
    # Plotare istoric
    print("\n[INFO] Plotare istoric...")
    plot_training_history(history, "training_history.png")
    
    print("\n" + "=" * 60)
    print("ANTRENAMENT COMPLETAT!")
    print("=" * 60)


def main():
    """Entry point"""
    train_model()


if __name__ == "__main__":
    main()
