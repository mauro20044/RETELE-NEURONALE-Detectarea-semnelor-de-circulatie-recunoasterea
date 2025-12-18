## Modul 1: Data Acquisition și Generare Date

**Scopul:** Generare și achiziție date originale pentru antrenament RN.

### Arhitectură

```
Input (date originale)
  ↓
14 imagini achiziții manual (cameră digitală)
  ↓
Augmentation (rotație, contrast, zgomot)
  ↓
16 imagini sintetice generate
  ↓
Output: 30 imagini total (14 originale + 16 augmentate)
```

### Utilizare

```python
# Generare imagini sintetice
from src.data_acquisition.synthetic_data import SyntheticDataGenerator

gen = SyntheticDataGenerator(output_dir='data/generated/')
gen.generate_samples(num_samples=16)
```

### Parametri

- **Dimensiune imagini:** 128×128 px (color/grayscale)
- **Format:** PNG
- **Augmentation:** Rotații ±30°, contrast [0.8 - 1.2], zgomot Gaussian σ=5
- **Output folder:** `data/labeled/` (organizat per clasă)

### Date Originale

**Total:** 14 imagini achiziții manual cu cameră digitală
- 1 imagine per clasă de semn de circulație
- Achiziție în condiții reale urbane
- Perspectivă naturală, iluminare variată

**Clasele:**
1. STOP
2. VITEZA_30
3. TRECERE_PIETONI
4. SCOALA
5. SENS_GIRATORIU
6. VIRAJA_DREAPTA
7. OCOLI_DREAPTA
8. FARA_PRIORITATE
9. PRIORITATE
10. CEDEAZA
11. CURBA_STANGA
12. ATENTIE_DREAPTA
13. OAMENI
14. AMBELE_SENSURI

### Fișiere Sursă

- `synthetic_data.py` – Generator imagini augmentate (Pillow)
- `__init__.py` – Import modul

### Output

- `data/labeled/[CLASE]/` – Imagini organizate per clasă
- `data/labels.txt` – Mapping clase index → eticheta

---

**Status:** ✅ Funcțional
