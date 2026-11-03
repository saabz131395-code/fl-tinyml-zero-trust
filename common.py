"""
Shared helpers: loading prepared data, building the autoencoder, and evaluation.
Used by the baseline, federated and TinyML scripts.
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import config as C


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_device(device):
    """Load one device's prepared data (made by 01_prepare_data.py)."""
    path = os.path.join(C.PROCESSED_DIR, f"{device}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 01_prepare_data.py first.")
    return dict(np.load(path, allow_pickle=False))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_autoencoder():
    """Small dense autoencoder: 115 -> 58 -> 29 -> 14 -> 29 -> 58 -> 115."""
    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(C.N_FEATURES,), name="traffic_features")
    x = inputs
    for units in C.HIDDEN_LAYERS:                    # encoder
        x = keras.layers.Dense(units, activation="relu")(x)
    for units in reversed(C.HIDDEN_LAYERS[:-1]):     # decoder
        x = keras.layers.Dense(units, activation="relu")(x)
    outputs = keras.layers.Dense(C.N_FEATURES, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="iot_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE), loss="mse")
    return model


def set_seed(seed=C.RANDOM_SEED):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def reconstruction_error(model, X):
    """Mean squared reconstruction error per sample (the anomaly score)."""
    X_hat = model.predict(X, batch_size=4096, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def pick_threshold(val_errors, percentile=C.THRESHOLD_PERCENTILE):
    """Threshold = given percentile of errors on benign validation data.
    Each device computes this locally from its own data."""
    return float(np.percentile(val_errors, percentile))


def per_attack_recall(errors, labels, threshold):
    """Detection rate for each attack type separately (plus FPR for benign)."""
    out = {}
    for name in np.unique(labels):
        mask = labels == name
        out[name] = float(np.mean(errors[mask] > threshold))
    return out


def evaluate(errors, y_true, threshold):
    """Detection metrics. Positive class = attack (1)."""
    y_pred = (errors > threshold).astype(np.int8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0          # = TPR, detection rate
    fpr = fp / (fp + tn) if fp + tn else 0.0             # benign flagged as attack
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    try:
        auc = roc_auc_score(y_true, errors)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "fpr": fpr, "f1": f1, "auc": auc, "threshold": threshold,
    }


def print_table(rows, title):
    """Pretty-print a list of result dicts."""
    print(f"\n{title}")
    print(f"{'Device':<13}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'FPR':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 61)
    for r in rows:
        print(f"{r['device']:<13}{r['accuracy']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['fpr']:>8.4f}{r['f1']:>8.4f}{r['auc']:>8.4f}")
