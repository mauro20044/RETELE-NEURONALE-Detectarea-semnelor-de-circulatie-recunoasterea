"""
Module de preprocesare pentru imagini de semne de circulație
"""
import os
import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImagePreprocessor:
    """Clasa pentru preprocesarea imaginilor"""
    
    def __init__(self, target_size=(128, 128), normalize=True):
        """
        Inițializare preprocessor
        
        Args:
            target_size: Dimensiunea țintă pentru imagini (înălțime, lățime)
            normalize: Dacă True, normalizează pixelii la [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def load_image(self, image_path):
        """
        Încarcă o imagine din fișier
        
        Args:
            image_path: Calea către fișierul imagine
            
        Returns:
            numpy array cu imaginea
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imaginea nu a fost găsită: {image_path}")
        
        try:
            img = Image.open(image_path).convert('L')
            return np.array(img)
        except Exception as e:
            raise ValueError(f"Nu s-a putut încărca imaginea: {image_path} - {e}")
    
    def resize_image(self, image, size=None):
        """
        Redimensionează o imagine
        
        Args:
            image: numpy array cu imaginea
            size: Noua dimensiune (înălțime, lățime). Dacă None, folosește target_size
            
        Returns:
            imagine redimensionată
        """
        if size is None:
            size = self.target_size
        
        img = Image.fromarray(image.astype(np.uint8))
        resized = img.resize((size[1], size[0]), Image.LANCZOS)
        return np.array(resized)
    
    def normalize_image(self, image):
        """
        Normalizează valorile pixelilor la [0, 1]
        
        Args:
            image: numpy array cu imaginea
            
        Returns:
            imagine normalizată
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def apply_histogram_equalization(self, image):
        """
        Aplică ecualizarea histogramei pentru a îmbunătăți contrastul
        
        Args:
            image: numpy array cu imaginea (0-255)
            
        Returns:
            imagine cu contrast îmbunătățit
        """
        if CV2_AVAILABLE:
            import cv2
            return cv2.equalizeHist(image.astype(np.uint8))
        else:
            # Fallback: manual histogram equalization cu PIL
            img_uint8 = image.astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            pil_img = ImageOps.equalize(pil_img)
            return np.array(pil_img)
    
    def preprocess(self, image_path, apply_equalization=True):
        """
        Preprocesează o imagine complet
        
        Args:
            image_path: Calea către imagine
            apply_equalization: Dacă True, aplică ecualizarea histogramei
            
        Returns:
            imagine preprocesată
        """
        # Încarcă imaginea
        img = self.load_image(image_path)
        
        # Aplică ecualizarea histogramei dacă este cerută
        if apply_equalization:
            img = self.apply_histogram_equalization(img)
        
        # Redimensionează
        img = self.resize_image(img)
        
        # Normalizează
        if self.normalize:
            img = self.normalize_image(img)
        
        return img
    
    def batch_preprocess(self, image_dir, output_dir, apply_equalization=True):
        """
        Preprocesează toate imaginile din directory
        
        Args:
            image_dir: Director cu imagini originale
            output_dir: Director pentru salvarea imaginilor preprocesate
            apply_equalization: Dacă True, aplică ecualizarea histogramei
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        processed_count = 0
        failed_count = 0
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    input_path = os.path.join(image_dir, filename)
                    output_path = os.path.join(output_dir, f"processed_{filename}")
                    
                    # Preprocesează imaginea
                    img = self.preprocess(input_path, apply_equalization)
                    
                    # Salvează imaginea preprocesată
                    if self.normalize:
                        img_to_save = (img * 255).astype(np.uint8)
                    else:
                        img_to_save = img.astype(np.uint8)
                    
                    pil_img = Image.fromarray(img_to_save)
                    pil_img.save(output_path)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Eroare la preprocesarea {filename}: {e}")
                    failed_count += 1
        
        print(f"Procesate: {processed_count}, Eșecuri: {failed_count}")
        return processed_count, failed_count


def load_dataset(data_dir, target_size=(128, 128)):
    """
    Încarcă toate imaginile dintr-un director
    
    Args:
        data_dir: Director cu imagini
        target_size: Dimensiunea țintă
        
    Returns:
        tuple (images, filenames) - array cu imagini și lista de nume de fișiere
    """
    images = []
    filenames = []
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
                img = np.array(img).astype(np.float32) / 255.0
                
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Eroare la încărcarea {filename}: {e}")
    
    return np.array(images), filenames
