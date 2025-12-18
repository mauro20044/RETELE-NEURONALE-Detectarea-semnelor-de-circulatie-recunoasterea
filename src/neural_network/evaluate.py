"""
Script evaluare model - Etapa 5
Încarcă model antrenat și evaluează pe test set
"""
import os
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import argparse
from src.preprocessing.image_preprocessing import ImagePreprocessor


def load_test_data(test_dir='data/test', labels_file='data/labels.txt'):
    """
    Încarcă imagini test din folder și labels din fișier
    
    Returns:
        x_test, y_test, label_names
    """
    print(f"\n[LOAD] Loading test images from {test_dir}...")
    
    # Load label mappings
    label_names = []
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            label_names = [line.strip() for line in f.readlines()]
    
    print(f"[INFO] Labels: {label_names}")
    
    # Load images
    x_test = []
    y_test = []
    files_list = []
    
    preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, filename)
            try:
                img = preprocessor.preprocess(img_path)
                x_test.append(img.flatten())
                
                # Infer label from filename or folder
                # For now: try to find label from filename
                label_idx = 0
                for idx, label_name in enumerate(label_names):
                    if label_name.lower() in filename.lower():
                        label_idx = idx
                        break
                
                y_test.append(label_idx)
                files_list.append(filename)
                print(f"  [OK] {filename} -> {label_names[label_idx] if label_idx < len(label_names) else 'UNKNOWN'}")
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")
    
    if len(x_test) == 0:
        print("[ERROR] No test images loaded!")
        return None, None, label_names, []
    
    return np.array(x_test), np.array(y_test), label_names, files_list


def evaluate_model(model_path, scaler_path, x_test, y_test, label_names, files_list=None):
    """
    Evaluează model antrenat pe test set
    """
    print(f"\n[LOAD] Loading model from {model_path}...")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None
    
    # Scale data
    x_test_scaled = scaler.transform(x_test)
    
    # Predict
    y_pred = model.predict(x_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n[EVALUATION RESULTS]")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Check Niveau 1 requirements
    print(f"\n[NIVEAU 1 REQUIREMENTS]")
    print(f"  Accuracy >= 0.65?  {accuracy:.4f} >= 0.65  {'✓ PASS' if accuracy >= 0.65 else '✗ FAIL'}")
    print(f"  F1-score >= 0.60?  {f1:.4f} >= 0.60  {'✓ PASS' if f1 >= 0.60 else '✗ FAIL'}")
    
    # Confusion matrix
    print(f"\n[CONFUSION MATRIX]")
    print(f"  True vs Predicted counts:")
    for i in range(min(cm.shape[0], 5)):
        print(f"    Class {i}: {cm[i]}")
    
    # Per-sample predictions
    if files_list:
        print(f"\n[PER-SAMPLE PREDICTIONS]")
        for fname, true_label, pred_label in zip(files_list, y_test, y_pred):
            true_name = label_names[true_label] if true_label < len(label_names) else f"CLASS_{true_label}"
            pred_name = label_names[pred_label] if pred_label < len(label_names) else f"CLASS_{pred_label}"
            match = "✓" if true_label == pred_label else "✗"
            print(f"    {match} {fname}: True={true_name}, Pred={pred_name}")
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'num_test_samples': len(y_test),
        'confusion_matrix': cm.tolist()
    }
    
    with open('results/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[SAVE] Metrics saved to results/test_metrics.json")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluare model antrenat')
    parser.add_argument('--model', default='models/trained_model.h5')
    parser.add_argument('--scaler', default='models/trained_scaler.pkl')
    parser.add_argument('--test_dir', default='data/test')
    parser.add_argument('--labels', default='data/labels.txt')
    args = parser.parse_args()
    
    # Load test data
    x_test, y_test, label_names, files_list = load_test_data(args.test_dir, args.labels)
    
    if x_test is None:
        print("[ERROR] Failed to load test data!")
        return
    
    print(f"\n[INFO] Loaded {len(x_test)} test samples")
    
    # Evaluate
    metrics = evaluate_model(args.model, args.scaler, x_test, y_test, label_names, files_list)
    
    if metrics:
        print(f"\n[COMPLETE] Evaluation finished!")
    else:
        print(f"\n[ERROR] Evaluation failed!")


if __name__ == '__main__':
    main()
