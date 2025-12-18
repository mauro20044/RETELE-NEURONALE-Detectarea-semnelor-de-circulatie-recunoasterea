"""
Screenshot tool - genereaza imagine demonstrativ a UI
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Creare imagine demonstrativ a UI-ului
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Background alb
ax.add_patch(Rectangle((0, 0), 12, 8, facecolor='white', edgecolor='none'))

# Header
ax.add_patch(FancyBboxPatch((0, 7), 12, 1, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#2C3E50', edgecolor='black', linewidth=2))
ax.text(6, 7.5, 'Traffic Sign Recognition System', fontsize=16, weight='bold', 
       color='white', ha='center', va='center')

# Divider
ax.plot([0, 12], [6.8, 6.8], 'k-', linewidth=2)

# Titlu upload form
ax.text(1, 6.3, 'Upload Image:', fontsize=12, weight='bold', ha='left')

# Simulare zona input
ax.add_patch(FancyBboxPatch((1, 4.5), 5, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#ECF0F1', edgecolor='#34495E', linewidth=2, linestyle='--'))
ax.text(3.5, 5.2, '📁 Choose File', fontsize=11, ha='center', va='center', style='italic')

# Buton upload
ax.add_patch(FancyBboxPatch((1, 3.8), 5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#27AE60', edgecolor='black', linewidth=2))
ax.text(3.5, 4.05, 'Upload & Predict', fontsize=11, weight='bold', 
       color='white', ha='center', va='center')

# Rezultate
ax.text(7.5, 6.3, 'Result:', fontsize=12, weight='bold', ha='left')

# Simulare imagine rezultat
ax.add_patch(Rectangle((6.5, 4), 5, 2, 
                       facecolor='#D5DBDB', edgecolor='black', linewidth=2))

# Desenez un simbol STOP simplu în dreptunghiul rezultat
stop_x, stop_y = 8.5, 5.5
octagon_points = np.array([
    [stop_x + 0.3, stop_y + 0.6],
    [stop_x + 0.6, stop_y + 0.6],
    [stop_x + 0.8, stop_y + 0.4],
    [stop_x + 0.8, stop_y + 0.1],
    [stop_x + 0.6, stop_y - 0.1],
    [stop_x + 0.3, stop_y - 0.1],
    [stop_x + 0.1, stop_y + 0.1],
    [stop_x + 0.1, stop_y + 0.4],
])

from matplotlib.patches import Polygon
octagon = Polygon(octagon_points, facecolor='red', edgecolor='white', linewidth=2)
ax.add_patch(octagon)
ax.text(stop_x + 0.45, stop_y + 0.25, 'STOP', fontsize=8, weight='bold', 
       color='white', ha='center', va='center')

# Rezultate text
ax.text(6.5, 3.7, 'Predicted Label:', fontsize=10, weight='bold', ha='left')
ax.add_patch(FancyBboxPatch((6.5, 3.2), 5, 0.4, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#D4EFDF', edgecolor='#27AE60', linewidth=2))
ax.text(9, 3.4, 'STOP', fontsize=11, weight='bold', color='#27AE60', ha='center', va='center')

ax.text(6.5, 2.9, 'Confidence:', fontsize=10, weight='bold', ha='left')
ax.add_patch(FancyBboxPatch((6.5, 2.4), 5, 0.4, 
                           boxstyle="round,pad=0.05", 
                           facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2))
ax.text(9, 2.6, '92.3%', fontsize=11, weight='bold', color='#F39C12', ha='center', va='center')

# Info box
ax.text(1, 3.3, 'Status: Ready', fontsize=9, style='italic', color='green')

# Footer
ax.text(6, 0.5, 'Powered by: Neural Network Model (14 classes) | Flask Web Service', 
       fontsize=8, ha='center', style='italic', color='gray')

plt.tight_layout()
plt.savefig('docs/screenshots/ui_demo.png', dpi=150, bbox_inches='tight', facecolor='white')
print("[OK] Screenshot UI salvat în: docs/screenshots/ui_demo.png")
plt.close()

print("Gata!")
