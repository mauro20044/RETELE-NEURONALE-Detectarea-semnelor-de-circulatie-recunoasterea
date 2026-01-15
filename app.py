from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from predict import load_model_file, load_labels, preprocess_image_file, predict_single
from src.preprocessing.image_preprocessing import ImagePreprocessor
import joblib

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# Prefer model optimizat (Etapa 6) dacă există; fallback pe modelul din Etapa 5
OPTIMIZED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'optimized_model.h5')
TRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.h5')

def resolve_model_path():
    """Returnează calea către modelul preferat (optimizat dacă există)."""
    return OPTIMIZED_MODEL_PATH if os.path.exists(OPTIMIZED_MODEL_PATH) else TRAINED_MODEL_PATH

MODEL_PATH = resolve_model_path()
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'trained_scaler.pkl')
# Prefer labels salvate lângă model; fallback la data/labels.txt
MODEL_LABELS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'labels.txt')
DEFAULT_LABELS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'labels.txt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prepare preprocessor, load model and labels at startup
preprocessor = ImagePreprocessor(target_size=(128,128), normalize=True)
model = None
scaler = None
try:
    model = load_model_file(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None
    print(f"[WARN] Could not load model at startup: {e}")

labels_path = MODEL_LABELS_PATH if os.path.exists(MODEL_LABELS_PATH) else DEFAULT_LABELS_PATH
labels = load_labels(labels_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    # save file
    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # If model not loaded, try to load now
    global model
    global scaler
    if model is None:
        try:
            model = load_model_file(resolve_model_path())
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            return render_template('index.html', error=f'No model available: {e}')

    # Preprocess uploaded file
    try:
        img = preprocess_image_file(preprocessor, save_path)
    except Exception as e:
        return render_template('index.html', error=f'Preprocessing failed: {e}')

    pred, conf, label = predict_single(model, img, labels, scaler=scaler)

    return render_template('result.html', filename=filename, label=label, confidence=f'{conf*100:.1f}%')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
