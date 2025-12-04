"""
Script interactiv pentru testare și demonstrație
"""
import os
import numpy as np
from PIL import Image
from src.neural_network.model import TrafficSignCNN
from src.preprocessing.image_preprocessing import load_dataset, ImagePreprocessor


def demo_menu():
    """Meniu demo interactiv"""
    print("\n" + "="*60)
    print("[DEMO] MENIU INTERACTIV - TESTARE MODEL")
    print("="*60)
    print("\n1. Testare pe imagini din train set")
    print("2. Testare pe imagini din test set")
    print("3. Testare pe o imagine specifică")
    print("4. Afișare statistici dataset")
    print("5. Antrenament model complet")
    print("6. Generare date sintetice")
    print("0. Ieșire\n")
    
    choice = input("Alege opțiune (0-6): ").strip()
    return choice


def test_train_set():
    """Testează pe imagini din train set"""
    print("\n[DEMO] TESTARE PE TRAIN SET")
    print("-" * 60)
    
    try:
        # Încarcă date
        x_train, filenames = load_dataset("data/train")
        print(f"Imagini încărcate: {len(x_train)}")
        
        # Crează și antrenează model
        model = TrafficSignCNN(num_classes=10)
        model.build_model()
        
        # Crează etichete dummy
        y_train = np.zeros((len(x_train), 10))
        for i in range(len(y_train)):
            y_train[i, 0] = 1
        
        print(f"Antrenament pe {len(x_train)} imagini...")
        history = model.train(x_train, y_train, x_train[:2], y_train[:2])
        
        # Fă predicții
        predictions = model.predict(x_train[:3])
        print(f"\nPredicții pe primele 3 imagini din train:")
        for i, (pred, fname) in enumerate(zip(predictions, filenames[:3])):
            print(f"  {i+1}. {fname}: {pred}")
        
    except Exception as e:
        print(f"Eroare: {e}")


def test_test_set():
    """Testează pe imagini din test set"""
    print("\n[DEMO] TESTARE PE TEST SET")
    print("-" * 60)
    
    try:
        # Încarcă date
        x_test, filenames = load_dataset("data/test")
        print(f"Imagini încărcate: {len(x_test)}")
        
        if len(x_test) == 0:
            print("Nu sunt imagini în test set!")
            return
        
        # Crează și antrenează model pe train
        x_train, _ = load_dataset("data/train")
        y_train = np.zeros((len(x_train), 10))
        for i in range(len(y_train)):
            y_train[i, 0] = 1
        
        model = TrafficSignCNN(num_classes=10)
        model.build_model()
        history = model.train(x_train, y_train, x_train[:2], y_train[:2])
        
        # Fă predicții pe test set
        predictions = model.predict(x_test)
        print(f"\nPredicții pe test set ({len(x_test)} imagini):")
        for i, (pred, fname) in enumerate(zip(predictions, filenames)):
            print(f"  {i+1}. {fname}: {pred}")
        
    except Exception as e:
        print(f"Eroare: {e}")


def test_specific_image():
    """Testează o imagine specifică"""
    print("\n[DEMO] TESTARE IMAGINE SPECIFICĂ")
    print("-" * 60)
    
    try:
        # Listează imagini disponibile
        train_dir = "data/train"
        images = [f for f in os.listdir(train_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print("Nu sunt imagini disponibile!")
            return
        
        print("\nImageni disponibile:")
        for i, img in enumerate(images[:5]):
            print(f"  {i+1}. {img}")
        if len(images) > 5:
            print(f"  ... și {len(images)-5} altele")
        
        choice = input("\nAlege indexul imaginii (1-5): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                image_file = os.path.join(train_dir, images[idx])
                
                # Preprocesează imaginea
                preprocessor = ImagePreprocessor()
                img = preprocessor.preprocess(image_file)
                
                print(f"\nImagine: {images[idx]}")
                print(f"Shape: {img.shape}")
                print(f"Dtype: {img.dtype}")
                print(f"Range: [{img.min():.3f}, {img.max():.3f}]")
                
            else:
                print("Index invalid!")
        except ValueError:
            print("Input invalid!")
            
    except Exception as e:
        print(f"Eroare: {e}")


def show_statistics():
    """Afișează statistici dataset"""
    print("\n[DEMO] STATISTICI DATASET")
    print("-" * 60)
    
    try:
        dirs = {
            "Train": "data/train",
            "Test": "data/test", 
            "Validation": "data/validation",
            "Raw": "data/raw",
            "Processed": "data/processed"
        }
        
        for name, path in dirs.items():
            if os.path.exists(path):
                images = [f for f in os.listdir(path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                size = sum(os.path.getsize(os.path.join(path, f)) 
                          for f in images) / 1024  # KB
                print(f"{name:12} : {len(images):3d} imagini ({size:8.1f} KB)")
            else:
                print(f"{name:12} : [NU EXISTA]")
        
        # Statistici model
        print("\n" + "-"*60)
        print("Model Antrenat:")
        model_path = "models/traffic_sign_model.h5"
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / 1024 / 1024
            print(f"  - Path: {model_path}")
            print(f"  - Dimensiune: {size:.2f} MB")
        else:
            print("  - Nu a fost antrenat")
            
    except Exception as e:
        print(f"Eroare: {e}")


def full_training():
    """Antrenament complet"""
    print("\n[DEMO] ANTRENAMENT COMPLET")
    print("-" * 60)
    
    try:
        # Încarcă date
        x_train, _ = load_dataset("data/train")
        x_val, _ = load_dataset("data/validation")
        x_test, _ = load_dataset("data/test")
        
        print(f"Date încărcate:")
        print(f"  - Train: {len(x_train)} imagini")
        print(f"  - Validation: {len(x_val)} imagini")
        print(f"  - Test: {len(x_test)} imagini")
        
        # Crează etichete
        num_classes = 10
        y_train = np.zeros((len(x_train), num_classes))
        for i in range(len(y_train)):
            y_train[i, 0] = 1
        y_val = np.zeros((len(x_val), num_classes))
        for i in range(len(y_val)):
            y_val[i, 0] = 1
        y_test = np.zeros((len(x_test), num_classes))
        for i in range(len(y_test)):
            y_test[i, 0] = 1
        
        # Antrenament
        model = TrafficSignCNN(num_classes=num_classes)
        model.build_model()
        model.compile_model()
        
        print("\nAntrenament în progres...")
        history = model.train(x_train, y_train, x_val, y_val, epochs=20)
        
        # Evaluare
        print("\nEvaluare pe test set...")
        loss, acc = model.evaluate(x_test, y_test)
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        # Salvează
        save = input("\nSalvează modelul? (y/n): ").strip().lower()
        if save == 'y':
            os.makedirs("models", exist_ok=True)
            model.save_model("models/traffic_sign_model.h5")
            print("Model salvat!")
        
    except Exception as e:
        print(f"Eroare: {e}")
        import traceback
        traceback.print_exc()


def generate_synthetic():
    """Generează date sintetice"""
    print("\n[DEMO] GENERARE DATE SINTETICE")
    print("-" * 60)
    
    try:
        from src.data_acquisition.synthetic_data import SyntheticDataGenerator
        
        count = input("Câte imagini per clasă? (default 5): ").strip()
        try:
            count = int(count) if count else 5
        except ValueError:
            count = 5
        
        print(f"\nGenerare {count * 4} imagini sintetice...")
        generator = SyntheticDataGenerator()
        generator.generate_dataset("data/synthetic", samples_per_class=count)
        
        print("✓ Date sintetice generate în data/synthetic/")
        
    except Exception as e:
        print(f"Eroare: {e}")


def main():
    """Meniu principal"""
    print("\n" + "="*60)
    print("[INFO] BINE VENIT LA DEMO INTERACTIV")
    print("="*60)
    
    while True:
        choice = demo_menu()
        
        if choice == '1':
            test_train_set()
        elif choice == '2':
            test_test_set()
        elif choice == '3':
            test_specific_image()
        elif choice == '4':
            show_statistics()
        elif choice == '5':
            full_training()
        elif choice == '6':
            generate_synthetic()
        elif choice == '0':
            print("\n[INFO] Ieșire... Revedere!")
            break
        else:
            print("\n[ERROR] Opțiune invalidă!")


if __name__ == "__main__":
    main()
