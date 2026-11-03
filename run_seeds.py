"""
Run the whole pipeline (steps 1-5) with several random seeds and report mean +/- std.

Each seed changes the data split, the attack samples and the model initialisation.
Results of each seed are copied to results/seed_<N>/ and combined into
results/multi_seed_summary.csv.

Run:  python run_seeds.py                    (seeds 42, 1, 2)
      python run_seeds.py --seeds 42 1 2 3 4
      python run_seeds.py --summary-only     (only rebuild the table from saved seeds)

Takes about 20-25 minutes per seed. Seeds already finished are skipped, so if the
computer stops, just run the same command again.
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
import pandas as pd

import config as C

STEPS = ["01_prepare_data.py", "02_baseline.py", "03_federated.py",
         "04_tinyml.py", "05_trust_engine.py"]
FILES = ["baseline_results.csv", "baseline_per_attack.csv", "fl_results.csv",
         "fl_per_attack.csv", "fl_history.csv", "tinyml_results.csv",
         "trust_engine_results.csv", "trust_false_quarantine.csv"]
METRICS = ["accuracy", "precision", "recall", "fpr", "f1", "auc"]
LABELS = {"local_only": "Local-only", "centralized": "Centralized",
          "federated": "Federated (FedAvg)", "tflite_int8": "Federated int8 (TinyML)"}


def run_seed(seed):
    out_dir = os.path.join(C.RESULTS_DIR, f"seed_{seed}")
    if os.path.exists(os.path.join(out_dir, "DONE")):
        print(f"Seed {seed}: already done, skipped")
        return
    env = dict(os.environ, SEED=str(seed), TF_CPP_MIN_LOG_LEVEL="2")
    for step in STEPS:
        print(f"Seed {seed}: running {step} ...", flush=True)
        r = subprocess.run([sys.executable, step], env=env, cwd=C.BASE_DIR,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-3000:])
            raise SystemExit(f"Seed {seed}: {step} failed (see error above)")
    os.makedirs(out_dir, exist_ok=True)
    for f in FILES:
        src = os.path.join(C.RESULTS_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, out_dir)
    open(os.path.join(out_dir, "DONE"), "w").close()
    print(f"Seed {seed}: finished, results in {out_dir}")


def summarize():
    rows, trust_rows, seeds = [], [], []
    for d in sorted(glob.glob(os.path.join(C.RESULTS_DIR, "seed_*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        seed = int(d.rsplit("_", 1)[1])
        seeds.append(seed)
        parts = [pd.read_csv(os.path.join(d, "baseline_results.csv")),
                 pd.read_csv(os.path.join(d, "fl_results.csv"))]
        tiny = pd.read_csv(os.path.join(d, "tinyml_results.csv"))
        parts.append(tiny[tiny["setting"] == "tflite_int8"])
        df = pd.concat(parts, ignore_index=True)
        avg = df.groupby("setting")[METRICS].mean().reset_index()
        avg["seed"] = seed
        rows.append(avg)

        t = pd.read_csv(os.path.join(d, "trust_engine_results.csv"))
        fq = pd.read_csv(os.path.join(d, "trust_false_quarantine.csv"))
        trust_rows.append({
            "seed": seed,
            "attacks_quarantined_pct": 100 * t["samples_to_quarantine"].notna().mean(),
            "mean_samples_to_quarantine": t["samples_to_quarantine"].mean(),
            "false_quarantine_pct": fq["pct_time_quarantined"].mean(),
        })

    if not rows:
        print("No finished seeds found.")
        return
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv(os.path.join(C.RESULTS_DIR, "multi_seed_all.csv"), index=False)

    g = allr.groupby("setting")[METRICS]
    mean, std = g.mean(), g.std(ddof=1).fillna(0)
    order = [s for s in LABELS if s in mean.index]
    table = pd.DataFrame({m: [f"{mean.loc[s, m]:.4f} ± {std.loc[s, m]:.4f}" for s in order]
                          for m in METRICS}, index=[LABELS[s] for s in order])
    table.to_csv(os.path.join(C.RESULTS_DIR, "multi_seed_summary.csv"))

    print(f"\nMean ± std over seeds {sorted(seeds)} (average of the 4 devices)")
    print(table.to_string())

    tr = pd.DataFrame(trust_rows)
    tr.to_csv(os.path.join(C.RESULTS_DIR, "multi_seed_trust.csv"), index=False)
    print("\nZero Trust engine per seed")
    print(tr.round(3).to_string(index=False))
    print(f"\nSaved: results/multi_seed_summary.csv, results/multi_seed_trust.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if not args.summary_only:
        for s in args.seeds:
            run_seed(s)
    summarize()


if __name__ == "__main__":
    main()
