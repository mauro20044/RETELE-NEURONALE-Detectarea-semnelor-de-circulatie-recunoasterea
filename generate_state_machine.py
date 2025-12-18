"""
State Machine Diagram - Traffic Sign Detection System
Based on user's design model
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(16, 12))
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Culori
color_normal = '#D4E8F7'
color_action = '#D4F7E8'
color_decision = '#FFF4D4'
color_error = '#F7D4D4'

def draw_state(ax, x, y, width, height, text, color='#D4E8F7', shape='rect'):
    """Draw a state box"""
    if shape == 'rect':
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.15", edgecolor='#333333', 
                            facecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold', 
            multialignment='center')
    return (x, y)

def draw_arrow(ax, x1, y1, x2, y2, label='', color='#333333', style='->', width=2.5):
    """Draw an arrow with label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=25, 
                           color=color, linewidth=width)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y + 0.3, label, fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='#999999', linewidth=1.5),
               color='#333333')

# Title
ax.text(8, 11.5, 'STATE MACHINE: Sistem Detecție Semnelor de Circulație', 
       fontsize=16, weight='bold', ha='center')
ax.text(8, 11.0, 'Detectare → Validare → Preprocess → Inferență → Decizie → Feedback', 
       fontsize=11, style='italic', ha='center', color='#666666')

# ============ TIER 1: START ============
y_t1 = 10.0
idle = draw_state(ax, 2, y_t1, 1.8, 0.9, 'IDLE\n(Camera Ready)', color_normal)

# Label
ax.text(2, y_t1 - 0.8, 'Așteptare', fontsize=9, ha='center', style='italic', color='#666666')

# ============ TIER 2: INPUT ============
y_t2 = 8.5
acquire = draw_state(ax, 2, y_t2, 1.8, 0.9, 'ACQUIRE_INPUT\n(Metadata +\nImage Capture)', color_action)

# Arrow: IDLE → ACQUIRE
draw_arrow(ax, idle[0], idle[1] - 0.45, acquire[0], acquire[1] + 0.45,
          'trigger upload /\ncamera ready', '#3366FF', '->', 2.5)

# Label
ax.text(2, y_t2 - 0.8, 'Captură + Metadate', fontsize=9, ha='center', style='italic', color='#666666')

# ============ TIER 3: VALIDATION ============
y_t3 = 7.0
validate = draw_state(ax, 2, y_t3, 1.8, 0.9, 'VALIDATE_FRAME\n(Format, Size,\nQuality Check)', color_decision)

# Arrow: ACQUIRE → VALIDATE
draw_arrow(ax, acquire[0], acquire[1] - 0.45, validate[0], validate[1] + 0.45,
          '', '#333333', '->', 2.5)

# Label
ax.text(2, y_t3 - 0.8, 'Validare Format', fontsize=9, ha='center', style='italic', color='#666666')

# ============ ERROR PATH (RIGHT) ============
y_error = 5.5
error_state = draw_state(ax, 5.5, y_error, 2.0, 0.9, 'ERROR\n(Blur/Lipsă/Invalid)', color_error)

# Arrow: VALIDATE → ERROR [cadru invalid]
draw_arrow(ax, validate[0] + 0.9, validate[1] - 0.2, error_state[0] - 1.0, error_state[1] + 0.2,
          'blur / lipsă\nmetadata', '#FF3333', '->', 2.5)

# Label
ax.text(5.5, y_error - 0.8, 'Eroare Captură', fontsize=9, ha='center', style='italic', color='#666666')

# ============ SUCCESS PATH (LEFT) ============
# PREPROCESS
y_preprocess = 5.5
preprocess = draw_state(ax, 2, y_preprocess, 1.8, 0.9, 'PREPROCESS\n(Resize, Normalize,\nHistogram Eq.)', color_action)

# Arrow: VALIDATE → PREPROCESS [cadru valid]
draw_arrow(ax, validate[0], validate[1] - 0.45, preprocess[0], preprocess[1] + 0.45,
          'cadru valid', '#33AA33', '->', 2.5)

# Label
ax.text(2, y_preprocess - 0.8, 'Normalizare 128×128', fontsize=9, ha='center', style='italic', color='#666666')

# ============ INFERENCE ============
y_inference = 4.0
inference = draw_state(ax, 2, y_inference, 1.8, 0.9, 'RUN_INFERENCE\n(Forward Pass CNN\nOutput: 14 classes)', color_action)

# Arrow: PREPROCESS → INFERENCE
draw_arrow(ax, preprocess[0], preprocess[1] - 0.45, inference[0], inference[1] + 0.45,
          'tensor ready', '#333333', '->', 2.5)

# Label
ax.text(2, y_inference - 0.8, 'Calcul RN Latență<100ms', fontsize=9, ha='center', style='italic', color='#666666')

# ============ DECISION NODE ============
y_decision = 2.5
decision = draw_state(ax, 2, y_decision, 1.8, 0.9, 'DECISION\n(argmax + Threshold\nScore >= Threshold?)', color_decision)

# Arrow: INFERENCE → DECISION
draw_arrow(ax, inference[0], inference[1] - 0.45, decision[0], decision[1] + 0.45,
          'scoruri CNN', '#333333', '->', 2.5)

# Label
ax.text(2, y_decision - 0.8, 'Analiză Output', fontsize=9, ha='center', style='italic', color='#666666')

# ============ SUCCESS OUTPUT ============
y_log = 1.0
log = draw_state(ax, 2, y_log, 1.8, 0.9, 'LOG_AND_FEEDBACK\n(Result + Timestamp\nSave to CSV)', color_action)

# Arrow: DECISION → LOG [scor >= threshold]
draw_arrow(ax, decision[0], decision[1] - 0.45, log[0], log[1] + 0.45,
          'scor >= threshold', '#33AA33', '->', 2.5)

# Label
ax.text(2, y_log - 0.8, 'Succes: Clasă Identificată', fontsize=9, ha='center', style='italic', color='#666666')

# ============ ERROR RECOVERY ============
recovery = draw_state(ax, 5.5, 1.0, 2.0, 0.9, 'RECOVERY\n(Retry / Notificare\nOperator)', color_error)

# Arrow: ERROR → RECOVERY
draw_arrow(ax, error_state[0], error_state[1] - 0.45, recovery[0], recovery[1] + 0.45,
          '', '#FF3333', '->', 2.5)

# Arrow: DECISION → RECOVERY [scor < threshold]
draw_arrow(ax, decision[0] + 0.9, decision[1] - 0.2, recovery[0] - 1.0, recovery[1] + 0.2,
          'model offline / iesire nevalidă\nscor < threshold', '#FF3333', '->', 2.5)

# Label
ax.text(5.5, 1.0 - 0.8, 'Eroare Decizie', fontsize=9, ha='center', style='italic', color='#666666')

# ============ CYCLE BACK PATHS ============
# LOG → IDLE (successful cycle)
draw_arrow(ax, log[0] - 0.9, log[1], 0.5, 10.0,
          'ciclu finalizat', '#3366FF', '->', 2.5)

# RECOVERY → IDLE (error handling)
draw_arrow(ax, recovery[0] + 1.0, recovery[1], 0.5, 10.0,
          'retry / notificare', '#FF3333', '->', 2.5)

# ============ RIGHT PANEL: DETAILS ============
details_x = 10.5
details_y = 10.0

# Parametri
ax.text(details_x, details_y, 'PARAMETRI SISTEM:', fontsize=12, weight='bold')

params = [
    '━━━━━━━━━━━━━━━━━━━━',
    '📷 INPUT:',
    '  • Format: JPG / PNG',
    '  • Dimensiune: 128×128 px',
    '  • Normalizare: [0, 1]',
    '',
    '🧠 MODEL (MLPClassifier):',
    '  • Layers: 16384 → 512 → 256 → 14',
    '  • Activation: ReLU + Softmax',
    '  • Optimizer: Adam',
    '  • Classes: 14 semnelor RO',
    '',
    '⚡ PERFORMANCE:',
    '  • Latență RN: < 100ms',
    '  • Ciclu total: < 200ms',
    '  • Confidence threshold: 0.7',
    '',
    '💾 OUTPUT:',
    '  • Label + Score',
    '  • Timestamp',
    '  • Saved to: CSV log',
]

y_param = details_y - 0.6
for param in params:
    ax.text(details_x, y_param, param, fontsize=8.5, family='monospace', va='top')
    y_param -= 0.4

# ============ BOTTOM: STATE DESCRIPTION ============
desc_y = 0.5
ax.text(8, desc_y, 'Flux: IDLE → ACQUIRE → VALIDATE → [SUCCESS: PREPROCESS → INFERENCE → DECISION → LOG] / [ERROR: RECOVERY] → IDLE', 
       fontsize=9, style='italic', ha='center',
       bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFACD', edgecolor='#999999', linewidth=1.5))

plt.tight_layout()
plt.savefig('docs/state_machine.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Diagrama State Machine (model user) salvată în: docs/state_machine.png")
plt.close()

print("✓ Diagramă creată cu succes!")
