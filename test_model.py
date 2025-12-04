"""
Script pentru testare completă a modelului și pipeline-ului
"""
import os
import sys
import numpy as np
from src.neural_network.model import TrafficSignCNN
from src.preprocessing.image_preprocessing import load_dataset, ImagePreprocessor
from src.data_acquisition.synthetic_data import SyntheticDataGenerator


def test_imports():
    """Test 1: Verifică dacă toate importurile funcționează"""
    print("[TEST 1] Verificare importuri...")
    try:
        from src.preprocessing.image_preprocessing import ImagePreprocessor, load_dataset
        from src.neural_network.model import TrafficSignCNN
        from src.neural_network.train import main as train_main
        from src.data_acquisition.synthetic_data import SyntheticDataGenerator
        print("  ✓ Toate importurile sunt OK\n")
        return True
    except Exception as e:
        print(f"  ✗ Eroare import: {e}\n")
        return False


def test_data_loading():
    """Test 2: Verifică dacă se încarcă corect datele"""
    print("[TEST 2] Verificare încărcare date...")
    try:
        train_dir = "data/train"
        if not os.path.exists(train_dir):
            print(f"  ✗ Directorul {train_dir} nu există\n")
            return False
        
        x_train, filenames = load_dataset(train_dir)
        print(f"  ✓ Încărcate {len(x_train)} imagini din train")
        print(f"    Shape: {x_train.shape}")
        print(f"    Min/Max: {x_train.min():.3f} / {x_train.max():.3f}\n")
        return True
    except Exception as e:
        print(f"  ✗ Eroare la încărcare date: {e}\n")
        return False


def test_image_preprocessing():
    """Test 3: Verifică funcțiile de preprocesare"""
    print("[TEST 3] Verificare preprocesare imagini...")
    try:
        preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
        
        # Test cu o imagine din train
        train_dir = "data/train"
        first_image = None
        for f in os.listdir(train_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                first_image = os.path.join(train_dir, f)
                break
        
        if first_image:
            img = preprocessor.preprocess(first_image)
            print(f"  ✓ Preprocesare imagine OK")
            print(f"    Shape: {img.shape}")
            print(f"    Dtype: {img.dtype}")
            print(f"    Range: [{img.min():.3f}, {img.max():.3f}]\n")
            return True
        else:
            print("  ✗ Nu s-a găsit imaginile în train\n")
            return False
            
    except Exception as e:
        print(f"  ✗ Eroare preprocesare: {e}\n")
        return False


def test_model_creation():
    """Test 4: Verifică crearea modelului"""
    print("[TEST 4] Verificare creație model...")
    try:
        model = TrafficSignCNN(input_shape=(128, 128, 1), num_classes=10)
        model.build_model()
        model.compile_model()
        
        print(f"  ✓ Model creat cu succes")
        summary = model.get_model_summary()
        if summary:
            print(f"    Tip: {type(summary).__name__}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Eroare creare model: {e}\n")
        return False


def test_model_prediction():
    """Test 5: Verifică predicțiile modelului"""
    print("[TEST 5] Verificare predicții model...")
    try:
        # Încarcă date
        x_train, _ = load_dataset("data/train")
        
        if len(x_train) == 0:
            print("  ✗ Nu s-au găsit imagini de antrenament\n")
            return False
        
        # Crează și antrenează model
        model = TrafficSignCNN(num_classes=10)
        model.build_model()
        
        # Crează etichete dummy
        y_train = np.zeros((len(x_train), 10))
        for i in range(len(y_train)):
            y_train[i, 0] = 1
        
        # Reshape pentru scikit-learn
        x_train_flat = x_train.reshape(len(x_train), -1)
        
        # Antrenaază
        print(f"  Antrenament pe {len(x_train)} imagini...")
        history = model.train(x_train, y_train, x_train[:2], y_train[:2])
        
        # Fă predicții
        predictions = model.predict(x_train[:2])
        print(f"  ✓ Predicții generate cu succes")
        print(f"    Shape predicții: {predictions.shape}")
        print(f"    Exemplu predicție: {predictions[0]}\n")
        return True
    except Exception as e:
        print(f"  ✗ Eroare predicție: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_saving():
    """Test 6: Verifică salvarea modelului"""
    print("[TEST 6] Verificare salvare model...")
    try:
        model = TrafficSignCNN(num_classes=10)
        model.build_model()
        
        # Creează director models dacă nu există
        os.makedirs("models", exist_ok=True)
        
        # Salvează
        test_path = "models/test_model.h5"
        model.save_model(test_path)
        
        if os.path.exists(test_path):
            size = os.path.getsize(test_path) / 1024 / 1024
            print(f"  ✓ Model salvat cu succes")
            print(f"    Cale: {test_path}")
            print(f"    Dimensiune: {size:.2f} MB\n")
            
            # Șterge fișierul de test
            os.remove(test_path)
            return True
        else:
            print("  ✗ Fișierul salvat nu a fost găsit\n")
            return False
    except Exception as e:
        print(f"  ✗ Eroare salvare: {e}\n")
        return False


def test_synthetic_data():
    """Test 7: Verifică generarea datelor sintetice"""
    print("[TEST 7] Verificare generare date sintetice...")
    try:
        generator = SyntheticDataGenerator()
        
        # Generează date în folder temporar
        test_dir = "data/synthetic_test"
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
        
        count = generator.generate_dataset(test_dir, samples_per_class=2)
        
        if count > 0:
            print(f"  ✓ Generate {count} imagini sintetice")
            print(f"    Locație: {test_dir}\n")
            
            # Șterge folder de test
            import shutil
            shutil.rmtree(test_dir)
            return True
        else:
            print("  ✗ Nu s-au generat imagini\n")
            return False
    except Exception as e:
        print(f"  ✗ Eroare generare date: {e}\n")
        return False


def test_full_pipeline():
    """Test 8: Verifică pipeline-ul complet"""
    print("[TEST 8] Verificare pipeline complet...")
    try:
        # Verifică directoarele necesare
        dirs = ["data/raw", "data/train", "data/test", "data/validation"]
        for d in dirs:
            if not os.path.exists(d):
                print(f"  ✗ Directorul {d} nu există\n")
                return False
        
        # Verifică dacă sunt imagini
        train_count = len([f for f in os.listdir("data/train") 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        test_count = len([f for f in os.listdir("data/test") 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        val_count = len([f for f in os.listdir("data/validation") 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"  ✓ Pipeline structurat corect")
        print(f"    Train: {train_count} imagini")
        print(f"    Test: {test_count} imagini")
        print(f"    Validation: {val_count} imagini\n")
        
        return train_count > 0 and test_count > 0 and val_count > 0
    except Exception as e:
        print(f"  ✗ Eroare pipeline: {e}\n")
        return False


def run_all_tests():
    """Rulează toate testele"""
    print("\n" + "="*60)
    print("[TESTING] SUITE COMPLET - RETEA NEURONALA")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_data_loading,
        test_image_preprocessing,
        test_model_creation,
        test_model_prediction,
        test_model_saving,
        test_synthetic_data,
        test_full_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Eroare în test: {e}\n")
            results.append(False)
    
    # Rezumat
    print("="*60)
    print("[REZUMAT] REZULTATE TESTE")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Teste trecute: {passed}/{total}")
    
    if passed == total:
        print("\n✓ TOATE TESTELE AU TRECUT!")
    else:
        print(f"\n✗ {total - passed} teste au eșuat")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
