"""
State Machine Diagram - Clear Layout with Proper Arrows
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(-1, 18)
ax.set_ylim(-0.5, 11)
ax.axis('off')

# Colors
c_idle = '#B3E5FC'
c_action = '#C8E6C9'
c_decision = '#FFE0B2'
c_error = '#FFCCCC'

def box(x, y, w, h, text, color):
    """Draw rounded rectangle state"""
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='#1a1a1a', facecolor=color, linewidth=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold', color='#1a1a1a')
    return (x, y)

def arrow(x1, y1, x2, y2, label='', color='#333333', lw=3):
    """Draw arrow with label"""
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle='->', mutation_scale=30,
                         color=color, linewidth=lw, zorder=5)
    ax.add_patch(arr)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.4, label, fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=color, linewidth=2),
               color=color, ha='center', zorder=10)

# Title
ax.text(9, 10.5, 'STATE MACHINE - Sistem Detecție Semnelor de Circulație', 
       fontsize=16, weight='bold', ha='center')

# ===== COLUMN 1: MAIN FLOW =====
col1_x = 2

# 1. IDLE
y_idle = 9
idle_state = box(col1_x, y_idle, 1.8, 0.8, 'IDLE\n(Ready)', c_idle)

# 2. ACQUIRE_INPUT
y_acq = 7.5
acq = box(col1_x, y_acq, 1.8, 0.9, 'ACQUIRE_INPUT\n(Capture +\nMetadata)', c_action)
arrow(col1_x, y_idle - 0.4, col1_x, y_acq + 0.45, 'start', '#0066CC', 3)

# 3. VALIDATE_FRAME
y_val = 5.8
val = box(col1_x, y_val, 1.8, 0.9, 'VALIDATE_FRAME\n(Format, Size,\nQuality)', c_decision)
arrow(col1_x, y_acq - 0.45, col1_x, y_val + 0.45, '', '#333333', 3)

# SUCCESS path (down)
y_prep = 4.0
prep = box(col1_x, y_prep, 1.8, 0.9, 'PREPROCESS\n(Resize 128×128\nNormalize)', c_action)
arrow(col1_x, y_val - 0.45, col1_x, y_prep + 0.45, 'VALID', '#00AA00', 3)

# 4. RUN_INFERENCE
y_inf = 2.3
inf = box(col1_x, y_inf, 1.8, 0.9, 'RUN_INFERENCE\n(Forward Pass\n14 Classes)', c_action)
arrow(col1_x, y_prep - 0.45, col1_x, y_inf + 0.45, '', '#333333', 3)

# 5. DECISION
y_dec = 0.5
dec = box(col1_x, y_dec, 1.8, 0.9, 'DECISION\n(argmax +\nThreshold?)', c_decision)
arrow(col1_x, y_inf - 0.45, col1_x, y_dec + 0.45, 'scores', '#333333', 3)

# ===== COLUMN 2: ERROR PATH =====
col2_x = 6.5

# ERROR STATE
y_error = 5.8
err = box(col2_x, y_error, 1.8, 0.8, 'ERROR\n(Invalid Frame)', c_error)
arrow(col1_x + 0.9, y_val, col2_x - 0.9, y_error, 'INVALID', '#CC0000', 3)

# RECOVERY
y_rec = 1.8
rec = box(col2_x, y_rec, 1.8, 0.8, 'RECOVERY\n(Notify + Retry)', c_error)
arrow(col2_x, y_error - 0.4, col2_x, y_rec + 0.4, '', '#CC0000', 3)

# Error back to IDLE (curved)
ax.annotate('', xy=(col1_x - 0.9, y_idle), xytext=(col2_x + 0.9, y_rec),
           arrowprops=dict(arrowstyle='->', lw=3, color='#CC0000',
                          connectionstyle="arc3,rad=0.5", zorder=5))
ax.text(5, 3.5, 'retry', fontsize=9, weight='bold', color='#CC0000',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CC0000', linewidth=2))

# ===== COLUMN 3: OUTPUT =====
col3_x = 11

# LOG_AND_FEEDBACK
y_log = 0.5
log = box(col3_x, y_log, 1.8, 0.8, 'LOG_AND_FEEDBACK\n(Save + Display)', c_action)
arrow(col1_x + 0.9, y_dec, col3_x - 0.9, y_log, 'score >= 0.7', '#00AA00', 3)

# LOW SCORE path
ax.annotate('', xy=(col2_x - 0.9, y_rec), xytext=(col1_x + 0.9, y_dec - 0.2),
           arrowprops=dict(arrowstyle='->', lw=3, color='#CC0000', zorder=5))
ax.text(3.5, 0.8, 'score < 0.7', fontsize=9, weight='bold', color='#CC0000',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CC0000', linewidth=2))

# Success back to IDLE
ax.annotate('', xy=(col1_x - 0.9, y_idle - 0.1), xytext=(col3_x + 0.9, y_log),
           arrowprops=dict(arrowstyle='->', lw=3, color='#0066CC',
                          connectionstyle="arc3,rad=-0.5", zorder=5))
ax.text(9, 7.5, 'done', fontsize=9, weight='bold', color='#0066CC',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#0066CC', linewidth=2))

# ===== RIGHT PANEL: DETAILED INFO =====
info_x = 13.5
info_y = 9.5

# Section 1: INPUT
ax.text(info_x, info_y, 'INPUT PARAMETERS:', fontsize=11, weight='bold')
params_in = [
    '• Format: JPG / PNG',
    '• Size: min 64×64 px',
    '• Quality: min 80%',
    '• Metadata: timestamp',
]
y = info_y - 0.5
for p in params_in:
    ax.text(info_x, y, p, fontsize=8.5)
    y -= 0.35

# Section 2: PROCESSING
y -= 0.2
ax.text(info_x, y, 'PROCESSING:', fontsize=11, weight='bold')
proc = [
    '• Target: 128×128 px',
    '• Normalize: [0, 1]',
    '• Histogram EQ: yes',
    '• Mode: Grayscale',
]
y -= 0.5
for p in proc:
    ax.text(info_x, y, p, fontsize=8.5)
    y -= 0.35

# Section 3: MODEL
y -= 0.2
ax.text(info_x, y, 'MODEL (MLPClassifier):', fontsize=11, weight='bold')
model_info = [
    '• Input: 16384 neurons',
    '• Hidden-1: 512',
    '• Hidden-2: 256',
    '• Output: 14 classes',
    '• Activation: ReLU→Softmax',
    '• Optimizer: Adam',
]
y -= 0.5
for p in model_info:
    ax.text(info_x, y, p, fontsize=8.5)
    y -= 0.35

# Section 4: THRESHOLDS
y -= 0.2
ax.text(info_x, y, 'PERFORMANCE:', fontsize=11, weight='bold')
thresh = [
    '• Confidence: >= 0.7',
    '• RN latency: < 100ms',
    '• Cycle time: < 200ms',
    '• Target accuracy: 85%+',
]
y -= 0.5
for p in thresh:
    ax.text(info_x, y, p, fontsize=8.5)
    y -= 0.35

# ===== LEGEND (BOTTOM LEFT) =====
leg_y = -0.3
ax.text(0.2, leg_y, 'LEGENDA:', fontsize=10, weight='bold')

leg_items = [
    (c_idle, 'Idle/Init', 1.5),
    (c_action, 'Action', 3.5),
    (c_decision, 'Decision', 5.5),
    (c_error, 'Error', 7.5),
]

for color, label, leg_x in leg_items:
    rect = FancyBboxPatch((leg_x - 0.35, leg_y - 0.25), 0.3, 0.25,
                         boxstyle='round,pad=0.02',
                         facecolor=color, edgecolor='#1a1a1a', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(leg_x + 0.5, leg_y - 0.125, label, fontsize=8.5, va='center', weight='bold')

plt.tight_layout()
plt.savefig('docs/state_machine.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[✓] Diagrama State Machine v2 creată: docs/state_machine.png")

