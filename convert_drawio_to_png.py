"""
Convert DrawIO file to PNG using matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Culori
color_normal = '#E8F4F8'
color_error = '#FFE6E6'
color_action = '#E8F8E8'
color_decision = '#FFF4E6'

# Funcție pentru a desena o stare
def draw_state(ax, x, y, width, height, text, color='#E8F4F8'):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                        boxstyle="round,pad=0.1", edgecolor='black', 
                        facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', wrap=True)
    return (x, y)

# Funcție pentru săgeți cu etichete
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           color=color, linewidth=2)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y + 0.3, label, fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               color=color, weight='bold')

# Title
ax.text(7, 9.5, 'STATE MACHINE: Sistem Detecție Semne Circulație', 
       fontsize=14, weight='bold', ha='center')

# Rând 1
y_row1 = 8.5
idle = draw_state(ax, 1, y_row1, 1.5, 0.8, 'IDLE\n(Așteptare)', color_normal)
wait = draw_state(ax, 3.5, y_row1, 1.8, 0.8, 'WAIT_UPLOAD\n(Formular)', color_normal)

# Arrow IDLE → WAIT_UPLOAD
draw_arrow(ax, idle[0] + 0.75, idle[1], wait[0] - 0.9, wait[1], 'user action', 'black')

# Rând 2
y_row2 = 6.8
receive = draw_state(ax, 1, y_row2, 1.8, 0.8, 'RECEIVE_FILE\n(Primire)', color_action)
validate = draw_state(ax, 3.5, y_row2, 1.8, 0.8, 'VALIDATE_IMAGE\n(Check)', color_decision)
error = draw_state(ax, 6, y_row2, 1.8, 0.8, 'ERROR_QUALITY\n(Nevalidă)', color_error)

# Arrows to row 2
draw_arrow(ax, wait[0], wait[1] - 0.4, receive[0] + 0.5, receive[1] + 0.4, '', 'blue')
draw_arrow(ax, receive[0] + 0.9, receive[1], validate[0] - 0.9, validate[1], '', 'black')

# Validate decision
draw_arrow(ax, validate[0] - 0.5, validate[1] - 0.4, receive[0], receive[1] - 0.4, '[Valid]', 'green')
draw_arrow(ax, validate[0] + 0.9, validate[1], error[0] - 0.9, error[1], '[Invalid]', 'red')

# Error back to wait
draw_arrow(ax, error[0], error[1] + 0.4, wait[0] + 0.5, wait[1] - 0.4, 'retry', 'red')

# Rând 3
y_row3 = 5.1
preprocess = draw_state(ax, 1, y_row3, 1.8, 0.8, 'PREPROCESS\n(Normalize)', color_action)
inference = draw_state(ax, 3.5, y_row3, 1.8, 0.8, 'RN_INFERENCE\n(Forward)', color_action)
get_pred = draw_state(ax, 6, y_row3, 1.8, 0.8, 'GET_PREDICTION\n(argmax)', color_action)

# Arrows to row 3
draw_arrow(ax, receive[0], receive[1] - 0.4, preprocess[0], preprocess[1] + 0.4, '', 'green')
draw_arrow(ax, preprocess[0] + 0.9, preprocess[1], inference[0] - 0.9, inference[1], '', 'black')
draw_arrow(ax, inference[0] + 0.9, inference[1], get_pred[0] - 0.9, get_pred[1], 'output', 'black')

# Rând 4
y_row4 = 3.4
display = draw_state(ax, 1, y_row4, 1.8, 0.8, 'DISPLAY_RESULT\n(UI output)', color_action)
log = draw_state(ax, 3.5, y_row4, 1.8, 0.8, 'LOG_PREDICTION\n(Save CSV)', color_action)
return_idle = draw_state(ax, 6, y_row4, 1.8, 0.8, 'RETURN_to_IDLE\n(Reset)', color_normal)

# Arrows to row 4
draw_arrow(ax, get_pred[0] - 0.5, get_pred[1] - 0.4, display[0] + 0.5, display[1] + 0.4, '< 200ms', 'black')
draw_arrow(ax, display[0] + 0.9, display[1], log[0] - 0.9, log[1], '', 'black')
draw_arrow(ax, log[0] + 0.9, log[1], return_idle[0] - 0.9, return_idle[1], '', 'black')

# Cycle back
draw_arrow(ax, return_idle[0], return_idle[1] - 0.4, wait[0], wait[1] - 0.4, 'cycle', 'blue')

# LEGENDA
legend_y = 1.8
ax.text(8.5, legend_y + 0.5, 'LEGENDA:', fontsize=11, weight='bold')

# Legend colors
legend_items = [
    (color_normal, 'Stare normală'),
    (color_action, 'Stare de acțiune'),
    (color_decision, 'Stare de decizie'),
    (color_error, 'Stare de eroare'),
]

for i, (color, label) in enumerate(legend_items):
    y_pos = legend_y - i * 0.35
    rect = FancyBboxPatch((8.5, y_pos - 0.12), 0.3, 0.25, 
                         boxstyle="round,pad=0.02", 
                         facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(9, y_pos, label, fontsize=9, va='center')

# PARAMETRI KRITICI
param_y = 1.8
ax.text(11.5, param_y + 0.5, 'PARAMETRI KRITICI:', fontsize=11, weight='bold')
params = [
    '• Input: JPG/PNG',
    '• Size: 128×128 px',
    '• RN latency: < 100ms',
    '• Output: label + confidence',
    '• Classes: 14 semne',
]

for i, param in enumerate(params):
    ax.text(11.5, param_y - i * 0.35, param, fontsize=8, va='center')

# Footer
ax.text(7, 0.2, 'Ciclu: Upload → Validate → Preprocess → RN → Display → Log → Idle (repetă)', 
       fontsize=9, style='italic', ha='center',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('docs/state_machine.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Diagrama State Machine (din DrawIO) salvată în: docs/state_machine.png")
plt.close()
