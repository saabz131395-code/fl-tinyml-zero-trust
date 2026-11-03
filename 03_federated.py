"""
Step 3: Federated learning with FedAvg (McMahan et al., 2017).

Each device (client) keeps its raw traffic data private. In every round:
  1. the server sends the current global model weights to all clients
  2. each client trains the model on its OWN data for a few epochs
  3. each client sends back only its updated WEIGHTS (never data)
  4. the server averages the weights, weighted by each client's data size

After training, each client picks its own threshold from its own benign
validation data and is evaluated on its own test data.

Optional differential privacy (DP-FedAvg, McMahan et al., 2018):
each client's update is clipped to a maximum size and the server adds
Gaussian noise to the average, so no single client's data can be recovered
from the model.

Run:
  python 03_federated.py                      (plain FedAvg)
  python 03_federated.py --dp-noise 0.5       (with differential privacy)

Output: results/fl_results*.csv, results/fl_per_attack*.csv,
        results/fl_history*.csv, models/fl_global_ae*.keras
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

import config as C
import common as U


def weights_to_vector(weights):
    return np.concatenate([w.ravel() for w in weights])


def vector_to_weights(vec, template):
    out, i = [], 0
    for w in template:
        out.append(vec[i:i + w.size].reshape(w.shape).astype(w.dtype))
        i += w.size
    return out


def fedavg(client_weights, client_sizes):
    """Plain FedAvg: weighted average of client weights by number of samples."""
    total = float(sum(client_sizes))
    return [sum(w[i] * (n / total) for w, n in zip(client_weights, client_sizes))
            for i in range(len(client_weights[0]))]


def dp_fedavg(global_weights, client_weights, clip, noise, rng):
    """DP-FedAvg: clip each client's update, average equally, add Gaussian noise."""
    g = weights_to_vector(global_weights)
    deltas = []
    for w in client_weights:
        d = weights_to_vector(w) - g
        norm = np.linalg.norm(d)
        deltas.append(d * min(1.0, clip / (norm + 1e-12)))
    avg = np.mean(deltas, axis=0)
    sigma = noise * clip / len(client_weights)
    avg = avg + rng.normal(0.0, sigma, size=avg.shape)
    return vector_to_weights(g + avg, global_weights)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=C.FL_ROUNDS)
    p.add_argument("--local-epochs", type=int, default=C.LOCAL_EPOCHS)
    p.add_argument("--dp-noise", type=float, default=0.0,
                   help="noise multiplier z for DP (0 = no differential privacy)")
    p.add_argument("--dp-clip", type=float, default=1.0,
                   help="max L2 norm of each client's update when DP is on")
    args = p.parse_args()

    use_dp = args.dp_noise > 0
    tag = f"_dp{args.dp_noise:g}" if use_dp else ""
    rng = np.random.default_rng(C.RANDOM_SEED)

    U.set_seed()
    data = {dev: U.load_device(dev) for dev in C.DEVICES}
    sizes = [len(data[d]["X_train"]) for d in C.DEVICES]

    # Server model + one persistent model per client (each keeps its own optimizer)
    server = U.build_autoencoder()
    clients = {dev: U.build_autoencoder() for dev in C.DEVICES}
    global_w = server.get_weights()
    n_params = server.count_params()

    mode = f"DP-FedAvg (noise={args.dp_noise}, clip={args.dp_clip})" if use_dp else "FedAvg"
    print(f"{mode}: {len(C.DEVICES)} clients, {args.rounds} rounds, "
          f"{args.local_epochs} local epochs, {n_params:,} parameters")
    print(f"{'Round':>5}{'Val loss':>12}{'Update norm (min/max)':>26}{'Time':>8}")
    print("-" * 51)

    history = []
    t_start = time.time()
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        client_w, norms = [], []
        for dev in C.DEVICES:
            m = clients[dev]
            m.set_weights(global_w)
            m.fit(data[dev]["X_train"], data[dev]["X_train"],
                  epochs=args.local_epochs, batch_size=C.BATCH_SIZE, verbose=0)
            w = m.get_weights()
            client_w.append(w)
            norms.append(np.linalg.norm(weights_to_vector(w) - weights_to_vector(global_w)))

        if use_dp:
            global_w = dp_fedavg(global_w, client_w, args.dp_clip, args.dp_noise, rng)
        else:
            global_w = fedavg(client_w, sizes)

        # Global model's validation loss on every client's benign validation data
        server.set_weights(global_w)
        val_losses = [float(np.mean(U.reconstruction_error(server, data[d]["X_val"])))
                      for d in C.DEVICES]
        val_loss = float(np.average(val_losses, weights=sizes))
        history.append({"round": rnd, "val_loss": val_loss,
                        "norm_min": min(norms), "norm_max": max(norms),
                        **{f"val_loss_{C.SHORT_NAMES[d]}": v for d, v in zip(C.DEVICES, val_losses)}})
        print(f"{rnd:>5}{val_loss:>12.5f}{min(norms):>13.3f} / {max(norms):<10.3f}{time.time() - t0:>7.0f}s")

    print(f"\nTotal training time: {time.time() - t_start:.0f}s")
    comm_mb = 2 * n_params * 4 * len(C.DEVICES) * args.rounds / 1e6
    print(f"Communication: {comm_mb:.1f} MB of weights exchanged in total (no raw data shared)")

    # ---------------- Evaluation (each client locally) ----------------
    results, attack_rows = [], []
    for dev in C.DEVICES:
        d = data[dev]
        threshold = U.pick_threshold(U.reconstruction_error(server, d["X_val"]))
        errors = U.reconstruction_error(server, d["X_test"])
        r = U.evaluate(errors, d["y_test"], threshold)
        r.update({"device": C.SHORT_NAMES[dev], "setting": "federated" + tag})
        results.append(r)
        per = U.per_attack_recall(errors, d["label_test"], threshold)
        per.update({"device": C.SHORT_NAMES[dev], "setting": "federated" + tag})
        attack_rows.append(per)

    U.print_table(results, f"C) FEDERATED results ({mode})")
    names = sorted(k for k in attack_rows[0] if k not in ("device", "setting", "benign"))
    print(f"\nDetection rate per attack type (federated{tag})")
    print(f"{'Attack':<18}" + "".join(f"{a['device']:>13}" for a in attack_rows))
    print("-" * (18 + 13 * len(attack_rows)))
    for n in names:
        print(f"{n:<18}" + "".join(f"{a[n]:>13.4f}" for a in attack_rows))

    server.save(os.path.join(C.MODELS_DIR, f"fl_global_ae{tag}.keras"))
    pd.DataFrame(results).to_csv(os.path.join(C.RESULTS_DIR, f"fl_results{tag}.csv"), index=False)
    pd.DataFrame(attack_rows).to_csv(os.path.join(C.RESULTS_DIR, f"fl_per_attack{tag}.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(C.RESULTS_DIR, f"fl_history{tag}.csv"), index=False)
    print(f"\nSaved results to: {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
