"""
Meniu SIMPLU si PRIETEN pentru INCEPATORI
"""
import os
import sys

def print_header():
    """Afișează header frumos"""
    print("\n" + "="*70)
    print(" "*15 + "SEMNE DE CIRCULATIE - MENIU PRINCIPAL")
    print("="*70)
    print()

def print_menu():
    """Afișează meniu"""
    print("Ce vrei sa faci?\n")
    print("  1. [RAPID] Testa daca totul merge (8 teste - 30 sec)")
    print("  2. [CAUTATOR] Vede statistici despre date")
    print("  3. [ANTRENOR] Antreneaza modelul de zero")
    print("  4. [EXPERIMENTATOR] Testeaza pe o imagine")
    print("  5. [EXPLORATOR] Meniu avansat (cum era inainte)")
    print("  0. IESIRE")
    print()

def option_1_quick_test():
    """Testare rapida"""
    print("\n" + "-"*70)
    print("TESTARE RAPIDA (8 teste)")
    print("-"*70)
    print("\nAstept... (aprox 30 secunde)\n")
    
    os.system("python test_model.py")
    
    input("\nApasa ENTER pentru a continua...")

def option_2_statistics():
    """Statistici dataset"""
    print("\n" + "-"*70)
    print("STATISTICI DATASET")
    print("-"*70)
    
    dirs = {
        "Imagini BRUTE (raw)": "data/raw",
        "Imagini PREPROCESATE": "data/processed",
        "Imagini ANTRENAMENT": "data/train",
        "Imagini TEST": "data/test",
        "Imagini VALIDARE": "data/validation"
    }
    
    print()
    for name, path in dirs.items():
        if os.path.exists(path):
            images = [f for f in os.listdir(path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            size_kb = sum(os.path.getsize(os.path.join(path, f)) 
                         for f in images) / 1024
            print(f"  {name:30} : {len(images):3d} imagini ({size_kb:8.1f} KB)")
        else:
            print(f"  {name:30} : [NU EXISTA]")
    
    # Model
    print()
    model_path = "models/traffic_sign_model.h5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"  {'MODEL ANTRENAT':30} : {size_mb:8.2f} MB")
    else:
        print(f"  {'MODEL ANTRENAT':30} : [NU EXISTA - ruleaza optiunea 3]")
    
    print()
    input("Apasa ENTER pentru a continua...")

def option_3_train():
    """Antrenament"""
    print("\n" + "-"*70)
    print("ANTRENAMENT MODEL")
    print("-"*70)
    print("""
ATENTIE!
- Prima data se PREPROCESAZA imaginile (40 sec)
- Apoi se ANTRENEAZA modelul (1-2 minute)
- Se salveaza modelul si un grafic

Total timp: ~3-5 MINUTE

Vrei sa continui? (y/n): """, end="")
    
    resp = input().strip().lower()
    if resp == 'y':
        print("\nStart... astept...\n")
        os.system("python main.py")
    else:
        print("Anulat.")
    
    input("\nApasa ENTER pentru a continua...")

def option_4_test_image():
    """Test pe imagine"""
    print("\n" + "-"*70)
    print("TESTEAZA PE O IMAGINE")
    print("-"*70)
    
    from src.preprocessing.image_preprocessing import load_dataset, ImagePreprocessor
    from src.neural_network.model import TrafficSignCNN
    import numpy as np
    
    try:
        # Afiseaza imagini disponibile
        train_dir = "data/train"
        images = [f for f in os.listdir(train_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print("\nERORE: Nu sunt imagini in data/train/")
            input("Apasa ENTER...")
            return
        
        print(f"\nAvailable imagini ({len(images)}):\n")
        for i, img in enumerate(images[:10]):
            print(f"  {i+1}. {img}")
        if len(images) > 10:
            print(f"  ... si {len(images)-10} altele")
        
        print(f"\nAlege un numar (1-{min(10, len(images))}): ", end="")
        try:
            idx = int(input().strip()) - 1
            if 0 <= idx < len(images):
                image_file = os.path.join(train_dir, images[idx])
                
                # Preprocesare
                preprocessor = ImagePreprocessor()
                img = preprocessor.preprocess(image_file)
                
                print(f"\n--- IMAGINE: {images[idx]} ---")
                print(f"Shape: {img.shape}")
                print(f"Tipul: {img.dtype}")
                print(f"Valori: min={img.min():.3f}, max={img.max():.3f}")
                print(f"Status: Preprocesata OK!")
                
            else:
                print("Numar invalid!")
        except ValueError:
            print("Introdu un numar valid!")
    
    except Exception as e:
        print(f"EROARE: {e}")
    
    input("\nApasa ENTER pentru a continua...")

def option_5_advanced():
    """Meniu avansat (vechi demo)"""
    print("\n" + "-"*70)
    print("MENIU AVANSAT (EXPERT)")
    print("-"*70)
    print()
    
    os.system("python demo.py")

def main():
    """Meniu principal"""
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print_header()
        print_menu()
        
        choice = input("Alege (0-5): ").strip()
        
        if choice == '1':
            option_1_quick_test()
        elif choice == '2':
            option_2_statistics()
        elif choice == '3':
            option_3_train()
        elif choice == '4':
            option_4_test_image()
        elif choice == '5':
            option_5_advanced()
        elif choice == '0':
            print("\nLa revedere! Succes cu proiectul! 🚀\n")
            break
        else:
            print("\nAlege o optiune valida! (0-5)")
            input("Apasa ENTER...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInchis de utilizator. La revedere!")
        sys.exit(0)
