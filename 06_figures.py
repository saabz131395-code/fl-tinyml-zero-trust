"""
Step 6: Make the figures and the summary table for the paper.

Reads the CSV files written by steps 2-4 and saves PNG figures (200 dpi) to
results/figures/ plus a combined summary table results/summary_table.csv.

Run:  python 06_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

FIG = os.path.join(C.RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"local_only": "#8c96a8", "centralized": "#e8a33d",
          "federated": "#2a6fdb", "tflite_int8": "#1f9d55"}
LABELS = {"local_only": "Local-only", "centralized": "Centralized",
          "federated": "Federated (FedAvg)", "tflite_int8": "Federated int8 (TinyML)"}
DEV_ORDER = [C.SHORT_NAMES[d] for d in C.DEVICES]


def read(name):
    path = os.path.join(C.RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  (missing {name}, skipped)")
        return None
    return pd.read_csv(path)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)


def main():
    base, fl, tiny = read("baseline_results.csv"), read("fl_results.csv"), read("tinyml_results.csv")
    frames = [f for f in (base, fl) if f is not None]
    if tiny is not None:
        frames.append(tiny[tiny["setting"] == "tflite_int8"])
    allres = pd.concat(frames, ignore_index=True)
    settings = [s for s in COLORS if s in set(allres["setting"])]

    # ---------- Figure 1: F1 per device and setting ----------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.8 / len(settings)
    x = np.arange(len(DEV_ORDER))
    for i, s in enumerate(settings):
        sub = allres[allres["setting"] == s].set_index("device").reindex(DEV_ORDER)
        ax.bar(x + (i - (len(settings) - 1) / 2) * w, sub["f1"], w,
               color=COLORS[s], label=LABELS[s])
    ax.set_xticks(x, DEV_ORDER)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Detection F1 per device")
    ax.legend(fontsize=8, frameon=False, ncol=len(settings), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    clean(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f1_comparison.png"), dpi=200)
    plt.close(fig)

    # ---------- Figure 2: FL convergence ----------
    hist = read("fl_history.csv")
    if hist is not None:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(hist["round"], hist["val_loss"], color=COLORS["federated"], lw=2,
                marker="o", ms=3, label="Weighted average")
        for c in hist.columns:
            if c.startswith("val_loss_"):
                ax.plot(hist["round"], hist[c], lw=1, alpha=0.6, label=c.replace("val_loss_", ""))
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation reconstruction loss")
        ax.set_title("Federated training convergence")
        ax.legend(fontsize=8, frameon=False)
        clean(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fl_convergence.png"), dpi=200)
        plt.close(fig)

    # ---------- Figure 3: per-attack detection heatmap (federated) ----------
    per = read("fl_per_attack.csv")
    if per is not None:
        per = per.set_index("device").reindex(DEV_ORDER)
        attacks = sorted(c for c in per.columns if c not in ("setting", "benign"))
        M = per[attacks].T.values
        fig, ax = plt.subplots(figsize=(6.5, 5))
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(DEV_ORDER)), DEV_ORDER)
        ax.set_yticks(range(len(attacks)), attacks)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "#222222")
        ax.set_title("Detection rate per attack type (federated)")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "per_attack_heatmap.png"), dpi=200)
        plt.close(fig)

    # ---------- Summary table ----------
    summary = (allres.groupby("setting")[["accuracy", "precision", "recall", "fpr", "f1", "auc"]]
               .mean().reindex(settings))
    summary.insert(0, "raw_data_shared", ["no" if s != "centralized" else "yes" for s in settings])
    summary.index = [LABELS[s] for s in settings]
    summary.to_csv(os.path.join(C.RESULTS_DIR, "summary_table.csv"))
    print("\nAverage over the 4 devices")
    print(summary.round(4).to_string())
    print(f"\nFigures saved to: {FIG}")


if __name__ == "__main__":
    main()
