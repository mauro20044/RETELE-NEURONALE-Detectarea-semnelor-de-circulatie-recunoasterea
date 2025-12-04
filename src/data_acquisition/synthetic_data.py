"""
Module pentru achiziția și generarea de date
"""
import os
import numpy as np
from PIL import Image, ImageDraw


class SyntheticDataGenerator:
    """Generator de date sintetice pentru semne de circulație"""
    
    def __init__(self, image_size=(128, 128)):
        """
        Inițializare generator
        
        Args:
            image_size: Dimensiunea imaginilor generate
        """
        self.image_size = image_size
    
    def generate_stop_sign(self):
        """Generează imaginea unui semn STOP"""
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Octagon roșu
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        radius = 50
        points = []
        for i in range(8):
            angle = i * np.pi / 4
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill='red', outline='black')
        draw.text((center[0]-20, center[1]-10), "STOP", fill='white')
        
        return np.array(img.convert('L'))
    
    def generate_yield_sign(self):
        """Generează imaginea unui semn YIELD"""
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Triunghi roșu
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        radius = 50
        points = [
            (center[0], center[1] - radius),
            (center[0] + radius, center[1] + radius),
            (center[0] - radius, center[1] + radius)
        ]
        
        draw.polygon(points, fill='red', outline='white')
        draw.text((center[0]-25, center[1]-10), "YIELD", fill='white')
        
        return np.array(img.convert('L'))
    
    def generate_speed_limit_sign(self, speed=50):
        """
        Generează imaginea unui semn de limită de viteză
        
        Args:
            speed: Valoarea limitei de viteză
        """
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Cerc roșu
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        radius = 50
        draw.ellipse(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            outline='red', width=3
        )
        draw.text((center[0]-20, center[1]-10), str(speed), fill='red')
        
        return np.array(img.convert('L'))
    
    def generate_no_entry_sign(self):
        """Generează imaginea unui semn NU INTRARE"""
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Cerc roșu cu bară albă
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        radius = 50
        draw.ellipse(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            fill='red', outline='darkred'
        )
        draw.rectangle(
            [center[0] - radius, center[1] - 10,
             center[0] + radius, center[1] + 10],
            fill='white'
        )
        
        return np.array(img.convert('L'))
    
    def generate_dataset(self, output_dir, samples_per_class=5):
        """
        Generează un dataset sintetic
        
        Args:
            output_dir: Director pentru salvare
            samples_per_class: Numărul de imagini pe clasă
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        generators = {
            'STOP': self.generate_stop_sign,
            'YIELD': self.generate_yield_sign,
            'SPEED': self.generate_speed_limit_sign,
            'NO_ENTRY': self.generate_no_entry_sign
        }
        
        count = 0
        for sign_name, generator in generators.items():
            for i in range(samples_per_class):
                img = generator()
                filename = os.path.join(output_dir, f"{sign_name}_{i:03d}.png")
                # Salvează cu PIL
                pil_img = Image.fromarray(img)
                pil_img.save(filename)
                count += 1
        
        print(f"Generate {count} imagini sintetice în {output_dir}")
        return count
