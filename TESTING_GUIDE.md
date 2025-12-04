# 🧪 Ghid Complet de Testare

## 📋 Moduri de Testare

### **1. Testare Automată - Suite Complet (RECOMAND)**
```bash
python test_model.py
```

Aceasta va rula 8 teste:
- ✓ Test 1: Verificare importuri
- ✓ Test 2: Încărcare date
- ✓ Test 3: Preprocesare imagini
- ✓ Test 4: Creație model
- ✓ Test 5: Predicții model
- ✓ Test 6: Salvare model
- ✓ Test 7: Generare date sintetice
- ✓ Test 8: Pipeline complet

**Output așteptat:**
```
============================================================
[TESTING] SUITE COMPLET - RETEA NEURONALA
============================================================
[TEST 1] Verificare importuri...
  ✓ Toate importurile sunt OK
...
Teste trecute: 8/8
✓ TOATE TESTELE AU TRECUT!
```

---

### **2. Testare Interactivă - Demo**
```bash
python demo.py
```

Meniu interactiv cu opțiuni:
1. Testare pe imagini din train set
2. Testare pe imagini din test set
3. Testare pe o imagine specifică
4. Afișare statistici dataset
5. Antrenament model complet
6. Generare date sintetice
0. Ieșire

---

### **3. Testare Pipeline Complet**
```bash
python main.py
```

Rulează întregul pipeline:
1. Preprocesare 16 imagini → 16 imagini normalize
2. Antrenament pe 11 imagini
3. Validare pe 2 imagini
4. Test pe 3 imagini
5. Salvare model (100 MB)
6. Generare grafic

**Output așteptat:**
```
[STEP 1] PREPROCESARE IMAGINI
Procesate: 16, Eșecuri: 0
[OK] Preprocesare completă: 16 imagini procesate

[STEP 2] ANTRENAMENT REȚEA NEURONALĂ
   Train: 11 imagini
   Test: 3 imagini
   Validation: 2 imagini

   Test Accuracy: 100.00%
```

---

### **4. Testare Individuală Pasului 1: Split Date**
```bash
python split_data.py
```

Distribuie imagini în train/test/validation:
```
Total imagini găsite: 16
Train: 11 imagini copiate
Test: 3 imagini copiate
Validation: 2 imagini copiate
```

---

### **5. Testare Pasului 2: Preprocesare**
```bash
python src/preprocessing/preprocess.py
```

Preprocesează imagini brute:
```
Procesate: 16, Eșecuri: 0
```

---

### **6. Testare Pasului 3: Antrenament**
```bash
python src/neural_network/train.py
```

Antenează modelul:
```
Train: 11 imagini
Validation: 2 imagini
Test: 3 imagini

Test Accuracy: 1.0000
```

---

## 🧬 Testare Cod Python Direct

### Exemplu 1: Testare Preprocesare
```python
from src.preprocessing.image_preprocessing import ImagePreprocessor
import os

preprocessor = ImagePreprocessor(target_size=(128, 128))

# Testează o imagine
image_path = "data/train/2627c2fe7f7552cbb5bbc881f6870d85.png"
img = preprocessor.preprocess(image_path)

print(f"Shape: {img.shape}")          # (128, 128)
print(f"Type: {img.dtype}")            # float32
print(f"Range: [{img.min()}, {img.max()}]")  # [0.0, 1.0]
```

### Exemplu 2: Testare Încărcare Date
```python
from src.preprocessing.image_preprocessing import load_dataset

x_train, filenames = load_dataset("data/train")

print(f"Imagini: {len(x_train)}")           # 11
print(f"Shape: {x_train.shape}")             # (11, 128, 128)
print(f"Fișiere: {filenames[:3]}")
```

### Exemplu 3: Testare Model
```python
from src.neural_network.model import TrafficSignCNN
from src.preprocessing.image_preprocessing import load_dataset
import numpy as np

# Încarcă date
x_train, _ = load_dataset("data/train")
y_train = np.zeros((len(x_train), 10))
y_train[:, 0] = 1

# Creează model
model = TrafficSignCNN(num_classes=10)
model.build_model()
model.compile_model()

# Antrenează
history = model.train(x_train, y_train, x_train[:2], y_train[:2])

# Predicții
predictions = model.predict(x_train[:3])
print(f"Predicții shape: {predictions.shape}")  # (3, 10)
```

### Exemplu 4: Testare Salvare/Încărcare
```python
from src.neural_network.model import TrafficSignCNN
import os

# Creează model
model1 = TrafficSignCNN()
model1.build_model()

# Salvează
os.makedirs("models", exist_ok=True)
model1.save_model("models/test.h5")

# Încarcă
model2 = TrafficSignCNN()
model2.load_model("models/test.h5")

print("Model salvat și încărcat cu succes!")
```

---

## 📊 Verificări de Integritate

### Verificare Structură
```bash
# Verifică dacă toate folderele și fișierele există
python -c "
import os
dirs = ['src', 'data', 'models', 'docs']
for d in dirs:
    exists = 'OK' if os.path.exists(d) else 'MISSING'
    print(f'{d}: {exists}')
"
```

### Verificare Imagini
```bash
# Contează imagini în fiecare director
python -c "
import os
for d in ['data/raw', 'data/train', 'data/test', 'data/validation']:
    count = len([f for f in os.listdir(d) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f'{d}: {count} imagini')
"
```

### Verificare Dependințe
```bash
# Verifică care pachete sunt instalate
python -c "
packages = ['numpy', 'PIL', 'sklearn', 'matplotlib']
for pkg in packages:
    try:
        __import__(pkg if pkg != 'PIL' else 'PIL')
        print(f'{pkg}: OK')
    except:
        print(f'{pkg}: MISSING')
"
```

---

## 🐛 Troubleshooting Teste

### Eroare: "No module named 'src'"
**Soluție:** Asigură-te că ești în directorul proiectului
```bash
cd path/to/RETELE-NEURONALE-Detectarea-semnelor-de-circulatie-recunoasterea
python test_model.py
```

### Eroare: "No images found"
**Soluție:** Ruleaza mai întâi distribuția datelor
```bash
python split_data.py
```

### Eroare: "Cannot load image"
**Soluție:** Verifică formatele de imagini
```bash
# Trebuie să fie PNG sau JPG
ls data/raw/*.{png,jpg,jpeg}
```

### Rețea prea lentă
**Soluție:** Reduce dimensiunea epocilor
```python
history = model.train(x_train, y_train, x_val, y_val, epochs=5)
```

---

## ✅ Checklist Testare

Înainte de a considera proiectul gata, verifică:

- [ ] `python test_model.py` - 8/8 teste TRECUT
- [ ] `python main.py` - Pipeline COMPLET cu succes
- [ ] 16 imagini în `data/raw/`
- [ ] 16 imagini preprocesate în `data/processed/`
- [ ] Model salvat în `models/traffic_sign_model.h5` (>50 MB)
- [ ] Grafic `training_history.png` generat
- [ ] `demo.py` funcționează (opțiune 4 - Statistici)
- [ ] Datele sunt distribuite corect:
  - [ ] 11 imagini în `data/train/`
  - [ ] 3 imagini în `data/test/`
  - [ ] 2 imagini în `data/validation/`

---

## 📈 Interpretarea Rezultatelor

### Accuracy 100%
- Modelul se potrivește perfect la date
- Normal pe dataset mic (16 imagini)
- Riscul overfitting

### Loss scăzut (~0.1)
- Modelul converge bine
- Predicții sigure

### Shape predicții (N, 10)
- N = numărul de imagini
- 10 = numărul de clase
- Corect!

---

## 💡 Sfaturi Testare

1. **Start cu `test_model.py`** - Verifică rapidă
2. **Apoi `demo.py`** - Testare interactivă
3. **Finally `main.py`** - Testare completă
4. Adaugă mai multe imagini în `data/raw/` pentru teste mai bune
5. Schimbă `num_classes` dacă ai mai multe tipuri de semne

---

