"""
Step 1: Prepare the N-BaIoT data for every device (client).

For each device:
  - benign traffic is shuffled and split into train (60%), validation (10%), test (30%)
  - a signed log transform compresses very large feature values (config.LOG_TRANSFORM)
  - a scaler is fitted on that device's OWN benign training data only
    (no data is shared between devices, as in federated learning)
  - attack traffic (Gafgyt + Mirai) is sampled and added to the test set only
  - everything is saved to data/processed/<device>.npz

Run:  python 01_prepare_data.py
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as C


def load_csv(path):
    X = pd.read_csv(path, dtype=np.float32).values
    if C.LOG_TRANSFORM:
        X = np.sign(X) * np.log1p(np.abs(X))   # compress very large feature values
    return X.astype(np.float32)


def prepare_device(device, rng):
    dev_dir = os.path.join(C.DATA_DIR, device)

    # ----- benign traffic -----
    benign = load_csv(os.path.join(dev_dir, "benign_traffic.csv"))
    if benign.shape[1] != C.N_FEATURES:
        raise ValueError(f"{device}: expected {C.N_FEATURES} columns, got {benign.shape[1]}")
    benign = benign[rng.permutation(len(benign))]

    n = len(benign)
    n_train = int(n * C.TRAIN_FRAC)
    n_val = int(n * C.VAL_FRAC)
    X_train = benign[:n_train]
    X_val = benign[n_train:n_train + n_val]
    X_benign_test = benign[n_train + n_val:]

    # ----- attack traffic (test only) -----
    attack_parts, attack_names = [], []
    for botnet, files in C.ATTACK_FILES.items():
        for f in files:
            path = os.path.join(dev_dir, botnet, f + ".csv")
            if not os.path.exists(path):
                print(f"   (missing {botnet}/{f}.csv, skipped)")
                continue
            data = load_csv(path)
            if len(data) > C.MAX_ROWS_PER_ATTACK:
                data = data[rng.choice(len(data), C.MAX_ROWS_PER_ATTACK, replace=False)]
            attack_parts.append(data)
            attack_names += [f"{botnet}_{f}"] * len(data)
    X_attack = np.vstack(attack_parts)

    # ----- scale using this device's benign TRAINING data only -----
    scaler = StandardScaler().fit(X_train)
    scale = lambda X: scaler.transform(X).astype(np.float32)

    X_test = np.vstack([scale(X_benign_test), scale(X_attack)])
    y_test = np.concatenate([np.zeros(len(X_benign_test), np.int8),
                             np.ones(len(X_attack), np.int8)])
    label_test = np.array(["benign"] * len(X_benign_test) + attack_names)

    out = os.path.join(C.PROCESSED_DIR, f"{device}.npz")
    np.savez_compressed(
        out,
        X_train=scale(X_train), X_val=scale(X_val),
        X_test=X_test, y_test=y_test, label_test=label_test,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )
    return len(X_train), len(X_val), len(X_benign_test), len(X_attack)


def main():
    rng = np.random.default_rng(C.RANDOM_SEED)
    print(f"{'Device':<14}{'Train':>9}{'Val':>8}{'Test benign':>13}{'Test attack':>13}{'Time':>8}")
    print("-" * 65)
    for device in C.DEVICES:
        t0 = time.time()
        tr, va, tb, ta = prepare_device(device, rng)
        print(f"{C.SHORT_NAMES[device]:<14}{tr:>9,}{va:>8,}{tb:>13,}{ta:>13,}{time.time() - t0:>7.0f}s")
    print(f"\nSaved to: {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
