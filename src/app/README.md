## Modul 3: Web Service / UI

**Scopul:** Interfață web pentru upload imagine și predicție automată a semnelor de circulație.

### Arhitectură

```
Frontend (HTML)
  ↓
POST /upload (form-data: image)
  ↓
Backend Flask (app.py)
  ├─ Receive & Validate image
  ├─ Preprocess (resize, normalize)
  ├─ RN Inference
  ├─ Format result
  ↓
Response HTML (result.html)
  ├─ Imagine afișată
  ├─ Eticheta semn
  └─ Confidence %
```

### API Endpoints

| Endpoint | Metodă | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | HTML cu formular upload |
| `/upload` | POST | `form: image (file)` | HTML cu rezultat + imagine |
| `/uploads/<filename>` | GET | `filename` | PNG image (static) |

### Utilizare

```bash
# Instalare dependințe
pip install flask

# Pornire server
python app.py

# Accesare în browser
http://127.0.0.1:5000/

# Incarcă imagine din data/test/
# Apasă "Upload & Predict"
# Primești eticheta și confidence în pagina următoare
```

### Fișiere

- `app.py` – Server Flask cu rute
- `templates/index.html` – Formular upload
- `templates/result.html` – Pagina rezultat
- `README.md` – Instrucțiuni (acest fișier)

### Configurare

```python
# Parametri server (in app.py)
app.run(host='0.0.0.0', port=5000, debug=True)

# Folder upload
UPLOAD_FOLDER = 'uploads/'
MAX_FILE_SIZE = 10 MB
```

### Flux Cerere

1. **User deschide** `http://127.0.0.1:5000/`
2. **Selectează imagine** din computer
3. **Apasă "Upload & Predict"**
4. **Server primește** POST request cu imagine
5. **Preprocesare:** resize la 128×128, normalize
6. **RN Inference:** forward pass → 14 clase
7. **Extrage etichetă** cu confidence maxim
8. **Returneaza** pagina result.html cu:
   - Imagine uploadată (displayed)
   - Eticheta semn (ex: "STOP")
   - Confidence % (ex: "92.3%")
   - Link pentru nouă predicție

### Timp Răspuns

- Upload fișier: < 1 sec
- Preprocess imagine: < 50ms
- RN Inference: < 100ms
- Render HTML: < 50ms
- **Total: < 200ms** (latență țintă)

### Status Etapa 4

- ✅ Server Flask funcționează pe port 5000
- ✅ Upload imagine funcționează
- ✅ RN inference se execută cu model reantrena
- ✅ Rezultatul se afișează corect în pagina result.html
- ✅ Screenshot demonstrativ: `docs/screenshots/ui_demo.png`

### Extensii Viitoare (Etapa 5+)

- [ ] Upload imagini multiple (batch prediction)
- [ ] Afișare confidence bar (visual)
- [ ] Stocare istoric predicții
- [ ] API endpoint pentru integrare sisteme externe
- [ ] Deployment cloud (AWS/Azure)

---

**Status:** ✅ Funcțional
