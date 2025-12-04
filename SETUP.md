# 📘 Ghid de Utilizare - Rețele Neuronale pentru Detectarea Semnelor de Circulație

## 📋 Structura Proiectului

```
project/
├── main.py                          # Script principal - rulează pipeline complet
├── split_data.py                    # Distribuie imagini în train/test/validation
├── transform.py                     # Preprocesare imagini
├── requirements.txt                 # Dependențe Python
├── data/
│   ├── raw/                         # Date brute
│   ├── processed/                   # Date preprocesate
│   ├── train/                       # Set de instruire (70%)
│   ├── test/                        # Set de testare (20%)
│   └── validation/                  # Set de validare (10%)
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py   # Clase pentru preprocesare
│   │   └── preprocess.py            # Script preprocesare
│   ├── neural_network/
│   │   ├── model.py                 # Arhitectura rețelei neuronale
│   │   └── train.py                 # Script de antrenament
│   └── data_acquisition/
│       └── synthetic_data.py        # Generator de date sintetice
├── models/                          # Modele salvate (creat la runtime)
└── docs/                            # Documentație

```

---

## 🚀 Instrucțiuni de Instalare

### 1. **Instalează Python și Git**
   - Python ≥ 3.8
   - Git pentru versionare

### 2. **Clonează Repository-ul**
```bash
git clone https://github.com/mauro20044/RETELE-NEURONALE-Detectarea-semnelor-de-circulatie-recunoasterea.git
cd RETELE-NEURONALE-Detectarea-semnelor-de-circulatie-recunoasterea
```

### 3. **Creează Environment Virtual (Recomandat)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. **Instalează Dependințe**
```bash
pip install -r requirements.txt
```

---

## 📖 Cum să Folosești

### Opțiunea 1: **Pipeline Complet (Recomandat)**
Rulează tot (preprocesare + antrenament) cu:
```bash
python main.py
```

### Opțiunea 2: **Pasuri Individuale**

#### Pasul 1: Distribuție Date
```bash
python split_data.py
```
Distribuie imaginile din `data/raw/` în `data/train/`, `data/test/`, `data/validation/` cu proporție 70-20-10.

#### Pasul 2: Preprocesare
```bash
python src/preprocessing/preprocess.py
```
Preprocesează imaginile (resize, normalizare, ecualizare histogramă).

#### Pasul 3: Antrenament Rețea
```bash
python src/neural_network/train.py
```
Antrenează modelul CNN pe datele pregătite.

---

## 🧠 Arhitectura Rețelei Neuronale

Modelul utilizat este o **Rețea Neuronală Convoluțională (CNN)** cu:

- **3 blocuri convoluționale** cu 32, 64 și 128 filtre
- **Batch Normalization** și **Dropout** pentru regularizare
- **Max Pooling** pentru reducerea dimensionalității
- **2 straturi Dense** cu 512 și 256 neuroni
- **Ieșire Softmax** pentru clasificare multi-clasă

### Hiperparametri
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 8
- **Epoci:** 50
- **Early Stopping:** Dacă val_loss nu se îmbunătățește 10 epoci

---

## 📊 Module și Funcții

### `src/preprocessing/image_preprocessing.py`
Clasa `ImagePreprocessor` cu metode:
- `load_image()` - Încarcă imaginea
- `resize_image()` - Redimensionează
- `normalize_image()` - Normalizează pixelii [0,1]
- `apply_histogram_equalization()` - Îmbunătățește contrast
- `preprocess()` - Pipeline complet pentru o imagine
- `batch_preprocess()` - Preprocesează director întreg

### `src/neural_network/model.py`
Clasa `TrafficSignCNN` cu metode:
- `build_model()` - Construiește arhitectura
- `compile_model()` - Compilează modelul
- `train()` - Antrenează pe date
- `evaluate()` - Evaluează pe test set
- `save_model()` / `load_model()` - Salvează/încarcă model

### `src/data_acquisition/synthetic_data.py`
Clasa `SyntheticDataGenerator`:
- Generează imagini sintetice de semne (STOP, YIELD, SPEED, NO_ENTRY)
- Util pentru testare și demonstrație

---

## 📈 Rezultate și Output

După rularea pipeline-ului, se vor genera:
- `models/traffic_sign_model.h5` - Model antrenat
- `training_history.png` - Grafic Accuracy și Loss
- Foldere `data/processed/`, `data/train/`, `data/test/`, `data/validation/`

---

## 🐛 Troubleshooting

### Eroare: "Module not found"
```bash
# Asigură-te că ești în directorul proiectului
cd RETELE-NEURONALE-Detectarea-semnelor-de-circulatie-recunoasterea
```

### Eroare: "No module named 'tensorflow'"
```bash
# Reinstalează dependințe
pip install --upgrade -r requirements.txt
```

### GPU Support (Optional - pentru accelerare)
```bash
# Instaleaza CUDA și cuDNN, apoi:
pip install tensorflow[and-cuda]
```

---

## 📝 Notă Importantă

- Poți adauga mai multe imagini brute în `data/raw/`
- Ruleaza din nou `python split_data.py` pentru a redistribui
- Parametrii modelului pot fi ajustati în `src/neural_network/model.py`
- Pentru clase noi, actualizeaza `num_classes` în `train.py`

---

## 📧 Contact

Proiect pentru Disciplina: **Rețele Neuronale**  
Institiție: **POLITEHNICA București – FIIR**

