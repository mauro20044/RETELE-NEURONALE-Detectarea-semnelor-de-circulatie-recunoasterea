## Modul 2: Neural Network

**Scopul:** Definire și antrenament model RN pentru clasificare semne circulație (14 clase).

### Arhitectură Model

```
Input Layer:
  Shape: (128, 128, 1) - imagine grayscale normalizată
  
Flatten:
  Output: 16384 neuroni
  
Dense Layer 1:
  Neuroni: 512
  Activation: ReLU
  
Dropout: 0.2
  
Dense Layer 2:
  Neuroni: 256
  Activation: ReLU
  
Dropout: 0.2
  
Output Layer:
  Neuroni: 14 (numărul de clase)
  Activation: Softmax
```

### Parametri Antrenament

- **Optimizer:** SGD (scikit-learn) / Adam (TensorFlow)
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy
- **Learning Rate:** 0.001
- **Batch Size:** 8
- **Epochs:** 50 (demo)
- **Validation Split:** 15%

### Utilizare

```python
from src.neural_network.model import TrafficSignCNN
import numpy as np

# Creare model
model = TrafficSignCNN(input_shape=(128, 128, 1), num_classes=14)
model.build_model()
model.compile_model()

# Antrenament
history = model.train(x_train, y_train, x_val, y_val, epochs=50, batch_size=8)

# Salvare model
model.save_model("models/traffic_sign_model.h5")

# Predicție
prediction = model.predict(image)
```

### Fișiere

- `model.py` – Definire clasa `TrafficSignCNN`
- `train.py` – Script antrenament și evaluare
- `__init__.py` – Import modul

### Implementare

**Fallback Strategy:**
1. **Prima opțiune:** TensorFlow/Keras (dacă instalat)
2. **Fallback:** scikit-learn `MLPClassifier` (garantat compatibil)

Aceasta asigură că modelul funcționează indiferent de環境.

### Output

- `models/traffic_sign_model.h5` – Model salvat (joblib pentru sklearn)
- `training_history.png` – Plot accuracy/loss curves
- Console output cu metrici evaluare

### Status Etapa 4

- ✅ Arhitectură definită și compilată
- ✅ Model reantrena cu 14 clase
- ✅ Salvat și reîncărcabil
- ✅ Funcțional pentru inferență

**Status Etapa 5:** Planificare antrenament avansat cu dataset mai mare pentru accuracy > 85%.

---

**Status:** ✅ Funcțional
