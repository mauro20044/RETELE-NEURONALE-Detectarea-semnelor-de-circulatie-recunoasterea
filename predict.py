"""
Simple inference script for traffic sign recognition
Usage examples:
  python predict.py --image data/test/2627c2fe7f7552cbb5bbc881f6870d85.png
  python predict.py --folder data/test --model models/traffic_sign_model.h5 --labels data/labels.txt

Notes:
- The project may use a scikit-learn MLP (saved with joblib) as fallback.
- If the model was trained with dummy labels, predictions are not meaningful; retrain with labeled images.
"""
import os
import argparse
import numpy as np
from PIL import Image

# try to import joblib for sklearn models
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# Try to detect TensorFlow support (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# Import project preprocessor
from src.preprocessing.image_preprocessing import ImagePreprocessor


def resolve_default_model():
    """Preferă models/optimized_model.h5 dacă există, altfel trained_model.h5."""
    base_dir = os.path.join(os.path.dirname(__file__), 'models')
    optimized = os.path.join(base_dir, 'optimized_model.h5')
    trained = os.path.join(base_dir, 'trained_model.h5')
    return optimized if os.path.exists(optimized) else trained


def load_labels(labels_path):
    if not labels_path:
        return None
    if not os.path.exists(labels_path):
        print(f"[WARN] Labels file not found: {labels_path}")
        return None
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def predict_single(model, img_array, labels=None, scaler=None):
    """Run prediction for one preprocessed image (128x128 normalized)"""
    # img_array expected shape (128,128) float in [0,1]
    if TENSORFLOW_AVAILABLE and hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        # TF model
        x = np.expand_dims(img_array, axis=(0, -1))  # (1,128,128,1)
        probs = model.predict(x)[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred])
    else:
        # sklearn model (expects flat input)
        x = img_array.reshape(1, -1)
        # Apply scaler if provided (for sklearn models trained with StandardScaler)
        if scaler is not None:
            x = scaler.transform(x)
        # some sklearn classifiers may not have predict_proba
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(x)[0]
            pred = int(np.argmax(probs))
            confidence = float(probs[pred])
        else:
            pred = int(model.predict(x)[0])
            confidence = 1.0
    label = labels[pred] if labels and pred < len(labels) else f"class_{pred}"
    return pred, confidence, label


def load_model_file(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try joblib load first
    if JOBLIB_AVAILABLE:
        try:
            m = joblib.load(model_path)
            print(f"[INFO] Loaded model with joblib from {model_path}")
            return m
        except Exception:
            pass

    # Try TensorFlow load
    if TENSORFLOW_AVAILABLE:
        try:
            m = keras.models.load_model(model_path)
            print(f"[INFO] Loaded TensorFlow model from {model_path}")
            return m
        except Exception:
            pass

    # fallback: try joblib even if not available above
    try:
        import joblib as _joblib
        m = _joblib.load(model_path)
        print(f"[INFO] Loaded model with joblib (fallback) from {model_path}")
        return m
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


def preprocess_image_file(preprocessor, path):
    img = preprocessor.preprocess(path)
    return img


def main():
    parser = argparse.ArgumentParser(description='Predict traffic sign(s) from images')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path to an image file')
    group.add_argument('--folder', help='Path to a folder with images')
    parser.add_argument('--model', default=None, help='Path to saved model (default: optimized_model.h5 dacă există, altfel trained_model.h5)')
    parser.add_argument('--labels', default='data/labels.txt', help='Optional labels file (one label per line)')
    parser.add_argument('--show', action='store_true', help='Show image(s) with prediction printed (opens window)')

    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else None
    if labels:
        print(f"[INFO] Loaded {len(labels)} labels from {args.labels}")

    chosen_model_path = args.model or resolve_default_model()
    print(f"[INFO] Loading model from: {chosen_model_path}")
    model = load_model_file(chosen_model_path)

    preprocessor = ImagePreprocessor(target_size=(128,128), normalize=True)

    paths = []
    if args.image:
        paths = [args.image]
    else:
        # list images in folder
        for f in os.listdir(args.folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(args.folder, f))
    if not paths:
        print("[ERROR] No images found to predict")
        return

    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] Skipping missing file: {p}")
            continue
        try:
            img = preprocess_image_file(preprocessor, p)
            pred, conf, label = predict_single(model, img, labels)
            print(f"{os.path.basename(p)} -> {label} (class {pred}) : {conf*100:.1f}%")
            if args.show:
                # display image and prediction
                im = Image.open(p).convert('RGB')
                im.show(title=f"{label} {conf*100:.1f}%")
        except Exception as e:
            print(f"[ERROR] {p} : {e}")

if __name__ == '__main__':
    main()
