"""
Step 2: Baselines (the reference results that federated learning is compared against).

  A) LOCAL-ONLY:  each device trains its own autoencoder on its own data only.
  B) CENTRALIZED: one autoencoder trained on all devices' data pooled together
                  (best accuracy, but needs raw data sent to one server, which
                  breaks privacy and Zero Trust; FL tries to match it without that).

Each device picks its own threshold from its own benign validation data.

Run:  python 02_baseline.py
Output: results/baseline_results.csv, results/baseline_per_attack.csv,
        models/centralized_ae.keras
"""
import os
import time
import numpy as np
import pandas as pd

import config as C
import common as U


def train(model, X_train, X_val):
    from tensorflow import keras
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                         restore_best_weights=True)
    hist = model.fit(X_train, X_train, validation_data=(X_val, X_val),
                     epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
                     callbacks=[stop], verbose=0)
    return len(hist.history["loss"])


def score_device(model, d, device, setting, attack_rows):
    threshold = U.pick_threshold(U.reconstruction_error(model, d["X_val"]))
    errors = U.reconstruction_error(model, d["X_test"])
    r = U.evaluate(errors, d["y_test"], threshold)
    r["device"] = C.SHORT_NAMES[device]
    r["setting"] = setting
    per = U.per_attack_recall(errors, d["label_test"], threshold)
    per.update({"device": C.SHORT_NAMES[device], "setting": setting})
    attack_rows.append(per)
    return r


def print_attack_table(attack_rows, setting):
    rows = [a for a in attack_rows if a["setting"] == setting]
    names = sorted(k for k in rows[0] if k not in ("device", "setting", "benign"))
    print(f"\nDetection rate per attack type ({setting})")
    print(f"{'Attack':<18}" + "".join(f"{r['device']:>13}" for r in rows))
    print("-" * (18 + 13 * len(rows)))
    for n in names:
        print(f"{n:<18}" + "".join(f"{r[n]:>13.4f}" for r in rows))


def main():
    U.set_seed()
    data = {dev: U.load_device(dev) for dev in C.DEVICES}
    results, attack_rows = [], []

    # ---------------- A) Local-only ----------------
    print("A) Local-only training (one model per device)...")
    local_rows = []
    for dev in C.DEVICES:
        t0 = time.time()
        model = U.build_autoencoder()
        epochs = train(model, data[dev]["X_train"], data[dev]["X_val"])
        r = score_device(model, data[dev], dev, "local_only", attack_rows)
        local_rows.append(r)
        print(f"   {C.SHORT_NAMES[dev]:<12} trained {epochs} epochs in {time.time() - t0:.0f}s")
    U.print_table(local_rows, "A) LOCAL-ONLY results")
    print_attack_table(attack_rows, "local_only")
    results += local_rows

    # ---------------- B) Centralized ----------------
    print("\nB) Centralized training (all devices' data pooled)...")
    t0 = time.time()
    X_train = np.vstack([data[d]["X_train"] for d in C.DEVICES])
    X_val = np.vstack([data[d]["X_val"] for d in C.DEVICES])
    model = U.build_autoencoder()
    epochs = train(model, X_train, X_val)
    print(f"   trained {epochs} epochs in {time.time() - t0:.0f}s on {len(X_train):,} rows")
    central_rows = []
    for dev in C.DEVICES:
        r = score_device(model, data[dev], dev, "centralized", attack_rows)
        central_rows.append(r)
    U.print_table(central_rows, "B) CENTRALIZED results")
    print_attack_table(attack_rows, "centralized")
    results += central_rows

    model.save(os.path.join(C.MODELS_DIR, "centralized_ae.keras"))
    print(f"\nModel size: {model.count_params():,} parameters")

    out = os.path.join(C.RESULTS_DIR, "baseline_results.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    out2 = os.path.join(C.RESULTS_DIR, "baseline_per_attack.csv")
    pd.DataFrame(attack_rows).to_csv(out2, index=False)
    print(f"Saved: {out}\nSaved: {out2}")


if __name__ == "__main__":
    main()
