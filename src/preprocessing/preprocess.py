"""
Script pentru preprocesarea imaginilor de semne de circulație
"""
import os
from src.preprocessing.image_preprocessing import ImagePreprocessor


def main():
    """Preprocesează imaginile din folderul raw"""
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    print("🔄 Inițializare preprocesare...")
    preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
    
    print(f"📁 Imaginile vor fi citite din: {raw_dir}")
    print(f"💾 Imaginile preprocesate vor fi salvate în: {processed_dir}")
    
    try:
        processed, failed = preprocessor.batch_preprocess(
            raw_dir, 
            processed_dir, 
            apply_equalization=True
        )
        
        print(f"\n✅ Preprocesare completă!")
        print(f"   - Imagini procesate: {processed}")
        print(f"   - Imagini eșuate: {failed}")
        
    except Exception as e:
        print(f"❌ Eroare: {e}")


if __name__ == "__main__":
    main()
