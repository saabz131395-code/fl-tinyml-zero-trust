"""
Step 5: Zero Trust engine driven by the on-device (int8 TinyML) anomaly detector.

Each device continuously proves it is not compromised. For every traffic sample:
  - the int8 model gives an anomaly score (reconstruction error)
  - the score is compared with the device's own threshold
  - the device's trust score is updated with an exponentially weighted average:
        trust = DECAY * trust + (1 - DECAY) * (0 if anomalous else 1)
  - the trust score decides the access level:
        trust >= 0.7          -> FULL access
        0.4 <= trust < 0.7    -> RESTRICTED (limited network access)
        trust < 0.4           -> QUARANTINED (isolated from the network)

For every device and attack type, a traffic stream is simulated:
  300 normal samples -> 100 attack samples -> 300 normal samples
and we measure how fast the device is restricted / quarantined, and how often
a device is wrongly quarantined during normal operation.

Run:  python 05_trust_engine.py
Output: results/trust_engine_results.csv, results/trust_false_quarantine.csv,
        results/figures/trust_timeline.png
"""
import os
import importlib
import numpy as np
import pandas as pd

import config as C
import common as U

DECAY = 0.9
FULL, RESTRICTED_BELOW, QUARANTINE_BELOW = 1.0, 0.7, 0.4
N_BEFORE, N_ATTACK, N_AFTER = 300, 100, 300


def level(trust):
    if trust < QUARANTINE_BELOW:
        return "QUARANTINED"
    if trust < RESTRICTED_BELOW:
        return "RESTRICTED"
    return "FULL"


def run_trust(anomalous):
    """Trust score over time for a sequence of anomaly flags (True/False)."""
    trust, out = FULL, []
    for a in anomalous:
        trust = DECAY * trust + (1 - DECAY) * (0.0 if a else 1.0)
        out.append(trust)
    return np.array(out)


def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) + 1 if len(idx) else None


def main():
    tiny = importlib.import_module("04_tinyml")
    path = os.path.join(C.MODELS_DIR, "fl_ae_int8.tflite")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 04_tinyml.py first.")
    with open(path, "rb") as f:
        model = tiny.TFLiteModel(f.read())

    rng = np.random.default_rng(C.RANDOM_SEED)
    rows, fq_rows, timelines = [], [], {}

    for dev in C.DEVICES:
        d = U.load_device(dev)
        name = C.SHORT_NAMES[dev]
        thr = U.pick_threshold(U.reconstruction_error(model, d["X_val"]))
        err = U.reconstruction_error(model, d["X_test"])
        labels = d["label_test"]
        benign_err = err[labels == "benign"]

        # False quarantine: run the trust engine over all normal test traffic
        t = run_trust(benign_err > thr)
        fq_rows.append({"device": name, "normal_samples": len(t),
                        "pct_time_restricted_or_worse": float(np.mean(t < RESTRICTED_BELOW)) * 100,
                        "pct_time_quarantined": float(np.mean(t < QUARANTINE_BELOW)) * 100})

        for attack in sorted(set(labels) - {"benign"}):
            att_err = err[labels == attack]
            before = rng.choice(benign_err, N_BEFORE, replace=False)
            during = rng.choice(att_err, N_ATTACK, replace=False)
            after = rng.choice(benign_err, N_AFTER, replace=False)
            stream = np.r_[before, during, after]
            trust = run_trust(stream > thr)

            seg = trust[N_BEFORE:N_BEFORE + N_ATTACK]
            post = trust[N_BEFORE + N_ATTACK:]
            rows.append({
                "device": name, "attack": attack,
                "samples_to_restricted": first_index(seg < RESTRICTED_BELOW),
                "samples_to_quarantine": first_index(seg < QUARANTINE_BELOW),
                "samples_to_recover_full": first_index(post >= RESTRICTED_BELOW),
                "min_trust_during_attack": float(seg.min()),
            })
            timelines[(name, attack)] = trust

    res = pd.DataFrame(rows)
    fq = pd.DataFrame(fq_rows)
    res.to_csv(os.path.join(C.RESULTS_DIR, "trust_engine_results.csv"), index=False)
    fq.to_csv(os.path.join(C.RESULTS_DIR, "trust_false_quarantine.csv"), index=False)

    # ---------------- Print summary ----------------
    print("Samples until the device is QUARANTINED after an attack starts ('-' = never)")
    pivot = res.pivot(index="attack", columns="device", values="samples_to_quarantine")
    pivot = pivot[[C.SHORT_NAMES[d] for d in C.DEVICES]]
    print(f"{'Attack':<18}" + "".join(f"{c:>13}" for c in pivot.columns))
    print("-" * (18 + 13 * len(pivot.columns)))
    for attack, r in pivot.iterrows():
        print(f"{attack:<18}" + "".join(f"{('-' if pd.isna(v) else int(v)):>13}" for v in r))

    print("\nWrong restrictions during NORMAL operation (false alarms of the trust engine)")
    print(f"{'Device':<13}{'Normal samples':>16}{'% restricted':>15}{'% quarantined':>16}")
    print("-" * 60)
    for _, r in fq.iterrows():
        print(f"{r['device']:<13}{r['normal_samples']:>16,}{r['pct_time_restricted_or_worse']:>15.3f}"
              f"{r['pct_time_quarantined']:>16.3f}")

    # ---------------- Figure: trust timeline ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(C.RESULTS_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    examples = [("Thermostat", "mirai_ack", "#2a6fdb"), ("Thermostat", "gafgyt_tcp", "#1f9d55"),
                ("BabyMonitor", "gafgyt_tcp", "#d94841")]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(N_BEFORE, N_BEFORE + N_ATTACK, color="#999999", alpha=0.15, label="attack period")
    ax.axhspan(0, QUARANTINE_BELOW, color="#d94841", alpha=0.06)
    ax.axhspan(QUARANTINE_BELOW, RESTRICTED_BELOW, color="#e8a33d", alpha=0.06)
    for dev, att, col in examples:
        if (dev, att) in timelines:
            ax.plot(timelines[(dev, att)], color=col, lw=1.6, label=f"{dev}: {att}")
    ax.axhline(RESTRICTED_BELOW, color="#e8a33d", ls="--", lw=1)
    ax.axhline(QUARANTINE_BELOW, color="#d94841", ls="--", lw=1)
    ax.text(5, 0.83, "FULL", fontsize=8, color="#555555")
    ax.text(5, 0.53, "RESTRICTED", fontsize=8, color="#555555")
    ax.text(5, 0.18, "QUARANTINED", fontsize=8, color="#555555")
    ax.set_xlabel("Traffic sample")
    ax.set_ylabel("Trust score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Zero Trust engine driven by the int8 federated anomaly detector")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(fig_dir, "trust_timeline.png")
    fig.savefig(out, dpi=200)
    print(f"\nSaved: results/trust_engine_results.csv, results/figures/trust_timeline.png")


if __name__ == "__main__":
    main()
