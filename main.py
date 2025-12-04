"""
Script pentru a rula întreaga pipeline: preprocesare + antrenament
"""
import os
import sys

# Adaugă src la path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """Verifică dacă sunt instalate dependințele necesare"""
    required_packages = ['sklearn', 'PIL', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("[ERROR] Lipsesc pachete Python:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstallează-le cu: python -m pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Execută pipeline-ul complet"""
    
    print("\n" + "=" * 60)
    print("[PIPELINE] PIPELINE COMPLET - SEMNE DE CIRCULAȚIE")
    print("=" * 60 + "\n")
    
    # Verifică dependințe
    if not check_dependencies():
        print("\n[ERROR] Nu putem continua fără dependințele necesare.")
        return
    
    # Import după verificarea dependințelor
    from src.preprocessing.image_preprocessing import ImagePreprocessor
    from src.neural_network.train import main as train_main
    
    # Pasul 1: Preprocesare
    print("[STEP 1] PREPROCESARE IMAGINI")
    print("-" * 60)
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    if os.path.exists(raw_dir) and os.listdir(raw_dir):
        try:
            preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
            processed, failed = preprocessor.batch_preprocess(
                raw_dir, 
                processed_dir, 
                apply_equalization=True
            )
            print(f"[OK] Preprocesare completă: {processed} imagini procesate\n")
        except Exception as e:
            print(f"[ERROR] Eroare la preprocesare: {e}\n")
            return
    else:
        print(f"[WARNING] Nu s-au găsit imagini în {raw_dir}\n")
    
    # Pasul 2: Antrenament
    print("[STEP 2] ANTRENAMENT REȚEA NEURONALĂ")
    print("-" * 60 + "\n")
    
    try:
        train_main()
    except Exception as e:
        print(f"[ERROR] Eroare la antrenament: {e}\n")
        return
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PIPELINE COMPLETAT CU SUCCES!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
