"""
Quick check: does a log transform help detect Gafgyt TCP/UDP floods?

For each device, trains a small local autoencoder twice:
  (a) current preprocessing: StandardScaler
  (b) log transform first:   sign(x) * log(1 + |x|), then StandardScaler
and compares detection of the flood attacks and the false positive rate.

Run:  python diagnose_floods.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config as C
import common as U

ATTACKS = ["gafgyt/tcp", "gafgyt/udp", "mirai/ack"]
N_ATTACK = 5000
EPOCHS = 15


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def run(X_tr, X_val, X_ben_test, attacks, use_log):
    from tensorflow import keras
    f = signed_log if use_log else (lambda X: X)
    sc = StandardScaler().fit(f(X_tr))
    t = lambda X: sc.transform(f(X)).astype(np.float32)
    m = U.build_autoencoder()
    m.fit(t(X_tr), t(X_tr), validation_data=(t(X_val), t(X_val)), epochs=EPOCHS,
          batch_size=C.BATCH_SIZE, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    thr = U.pick_threshold(U.reconstruction_error(m, t(X_val)))
    e_ben = U.reconstruction_error(m, t(X_ben_test))
    out = {"FPR": float(np.mean(e_ben > thr))}
    for name, X_att in attacks.items():
        e_att = U.reconstruction_error(m, t(X_att))
        out[name] = float(np.mean(e_att > thr))
        y = np.r_[np.zeros(len(e_ben)), np.ones(len(e_att))]
        out[name + " AUC"] = float(roc_auc_score(y, np.r_[e_ben, e_att]))
    return out


def main():
    U.set_seed()
    rng = np.random.default_rng(C.RANDOM_SEED)
    rows = []
    for dev in C.DEVICES:
        d = os.path.join(C.DATA_DIR, dev)
        ben = pd.read_csv(os.path.join(d, "benign_traffic.csv"), dtype=np.float32).values
        ben = ben[rng.permutation(len(ben))]
        n = len(ben)
        a, b = int(n * C.TRAIN_FRAC), int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
        attacks = {}
        for at in ATTACKS:
            X = pd.read_csv(os.path.join(d, at + ".csv"), dtype=np.float32).values
            attacks[at.replace("/", "_")] = X[rng.choice(len(X), min(N_ATTACK, len(X)), replace=False)]
        for use_log in (False, True):
            r = run(ben[:a], ben[a:b], ben[b:], attacks, use_log)
            r.update({"device": C.SHORT_NAMES[dev], "preprocessing": "log+standard" if use_log else "standard"})
            rows.append(r)
            print(f"{C.SHORT_NAMES[dev]:<12} {r['preprocessing']:<13} FPR={r['FPR']:.4f}  "
                  f"tcp={r['gafgyt_tcp']:.4f}  udp={r['gafgyt_udp']:.4f}  "
                  f"mirai_ack={r['mirai_ack']:.4f}  (tcp AUC={r['gafgyt_tcp AUC']:.4f})")
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "diagnose_floods.csv"), index=False)


if __name__ == "__main__":
    main()
