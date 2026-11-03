"""
Step 4 (TinyML): shrink the federated model for a microcontroller and measure the cost.

  1. converts the federated global model to TensorFlow Lite:
       - float32 TFLite (reference)
       - full int8 TFLite (what runs on an ESP32 / ARM Cortex-M with TFLite Micro)
  2. measures model size, inference time per sample, and detection accuracy
     of the int8 model on every device's test data
  3. writes the int8 model as a C header (models/fl_ae_int8.h) that can be
     compiled into ESP32 / Arduino firmware with TFLite Micro

Run:  python 04_tinyml.py                       (uses models/fl_global_ae.keras)
      python 04_tinyml.py --model models/centralized_ae.keras

Output: results/tinyml_results.csv, results/tinyml_per_attack.csv,
        models/fl_ae_fp32.tflite, models/fl_ae_int8.tflite, models/fl_ae_int8.h
"""
import os
import time
import shutil
import argparse
import numpy as np
import pandas as pd

import config as C
import common as U


# ---------------------------------------------------------------------------
# TFLite helpers
# ---------------------------------------------------------------------------
def get_interpreter_class():
    """TFLite interpreter moved to the 'ai-edge-litert' package in newer TF versions."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter


def convert(model, rep_data, int8):
    import tensorflow as tf
    export_dir = os.path.join(C.MODELS_DIR, "_savedmodel_tmp")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    model.export(export_dir)
    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative():
            for i in range(0, min(len(rep_data), 2000)):
                yield [rep_data[i:i + 1]]

        conv.representative_dataset = representative
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    tfl = conv.convert()
    shutil.rmtree(export_dir, ignore_errors=True)
    return tfl


class TFLiteModel:
    """Runs a .tflite model on batches, handling int8 quantization of input/output."""

    def __init__(self, tflite_bytes, batch=4096):
        Interpreter = get_interpreter_class()
        self.tfl = tflite_bytes
        self.batch = batch
        self.it = Interpreter(model_content=tflite_bytes)
        inp = self.it.get_input_details()[0]
        self.it.resize_tensor_input(inp["index"], [batch, C.N_FEATURES])
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def _q(self, X, d):
        if d["dtype"] == np.int8:
            s, z = d["quantization"]
            return np.clip(np.round(X / s + z), -128, 127).astype(np.int8)
        return X.astype(np.float32)

    def _dq(self, Y, d):
        if d["dtype"] == np.int8:
            s, z = d["quantization"]
            return (Y.astype(np.float32) - z) * s
        return Y

    def predict(self, X, **_):
        out = np.empty_like(X, dtype=np.float32)
        for i in range(0, len(X), self.batch):
            chunk = X[i:i + self.batch]
            n = len(chunk)
            if n < self.batch:
                chunk = np.vstack([chunk, np.zeros((self.batch - n, X.shape[1]), np.float32)])
            self.it.set_tensor(self.inp["index"], self._q(chunk, self.inp))
            self.it.invoke()
            out[i:i + n] = self._dq(self.it.get_tensor(self.out["index"]), self.out)[:n]
        return out


def latency_ms(tflite_bytes, X, n=1000):
    """Average time for ONE sample (batch size 1), like on a real device."""
    Interpreter = get_interpreter_class()
    it = Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    helper = TFLiteModel.__new__(TFLiteModel)
    samples = [helper._q(X[i:i + 1], inp) for i in range(n)]
    for s in samples[:50]:                      # warm-up
        it.set_tensor(inp["index"], s); it.invoke()
    t0 = time.perf_counter()
    for s in samples:
        it.set_tensor(inp["index"], s); it.invoke(); it.get_tensor(out["index"])
    return (time.perf_counter() - t0) / n * 1000


def peak_activation_bytes(bytes_per_value):
    """Peak activation memory for one sample: the largest (input + output) pair of
    consecutive layers, since TFLite Micro reuses buffers between layers.
    The real tensor arena on a board adds a few KB of overhead on top of this."""
    sizes = [C.N_FEATURES] + C.HIDDEN_LAYERS + list(reversed(C.HIDDEN_LAYERS[:-1])) + [C.N_FEATURES]
    return max(a + b for a, b in zip(sizes, sizes[1:])) * bytes_per_value


def write_c_header(tflite_bytes, path, name="fl_ae_int8"):
    lines = [f"// Federated autoencoder for IoT anomaly detection (int8, TFLite Micro)",
             f"// Size: {len(tflite_bytes)} bytes", "#pragma once",
             f"alignas(16) const unsigned char {name}_tflite[] = {{"]
    for i in range(0, len(tflite_bytes), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 12]) + ",")
    lines += ["};", f"const unsigned int {name}_tflite_len = {len(tflite_bytes)};", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
def main():
    from tensorflow import keras
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(C.MODELS_DIR, "fl_global_ae.keras"))
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"{args.model} not found. Run 03_federated.py first.")
    model = keras.models.load_model(args.model)
    data = {dev: U.load_device(dev) for dev in C.DEVICES}

    # Representative data for int8 calibration: benign training rows from every client.
    # (In a real deployment each client would calibrate on its own data.)
    rng = np.random.default_rng(C.RANDOM_SEED)
    rep = np.vstack([data[d]["X_train"][rng.choice(len(data[d]["X_train"]), 500, replace=False)]
                     for d in C.DEVICES]).astype(np.float32)
    rep = rep[rng.permutation(len(rep))]

    print("Converting to TensorFlow Lite...")
    fp32 = convert(model, rep, int8=False)
    int8 = convert(model, rep, int8=True)
    with open(os.path.join(C.MODELS_DIR, "fl_ae_fp32.tflite"), "wb") as f:
        f.write(fp32)
    with open(os.path.join(C.MODELS_DIR, "fl_ae_int8.tflite"), "wb") as f:
        f.write(int8)
    write_c_header(int8, os.path.join(C.MODELS_DIR, "fl_ae_int8.h"))

    keras_kb = model.count_params() * 4 / 1024
    sample = data[C.DEVICES[0]]["X_test"]
    lat32 = latency_ms(fp32, sample)
    lat8 = latency_ms(int8, sample)

    print(f"\n{'Model':<16}{'Size (KB)':>11}{'Latency/sample (ms, PC)':>26}{'Peak act. (B)':>15}")
    print("-" * 68)
    print(f"{'Keras float32':<16}{keras_kb:>11.1f}{'-':>26}{'-':>15}")
    print(f"{'TFLite float32':<16}{len(fp32) / 1024:>11.1f}{lat32:>26.4f}{peak_activation_bytes(4):>15,}")
    print(f"{'TFLite int8':<16}{len(int8) / 1024:>11.1f}{lat8:>26.4f}{peak_activation_bytes(1):>15,}")
    print(f"\nint8 model is {len(fp32) / len(int8):.1f}x smaller than float32 TFLite.")

    # ---------------- Accuracy: Keras vs int8 TFLite ----------------
    tfl8 = TFLiteModel(int8)
    results, attack_rows = [], []
    for name, m in (("keras_fp32", model), ("tflite_int8", tfl8)):
        rows = []
        for dev in C.DEVICES:
            d = data[dev]
            thr = U.pick_threshold(U.reconstruction_error(m, d["X_val"]))
            err = U.reconstruction_error(m, d["X_test"])
            r = U.evaluate(err, d["y_test"], thr)
            r.update({"device": C.SHORT_NAMES[dev], "setting": name})
            rows.append(r)
            per = U.per_attack_recall(err, d["label_test"], thr)
            per.update({"device": C.SHORT_NAMES[dev], "setting": name})
            attack_rows.append(per)
        U.print_table(rows, f"{name} detection results")
        results += rows

    summary = pd.DataFrame(results)
    summary["model_kb"] = summary["setting"].map({"keras_fp32": keras_kb, "tflite_int8": len(int8) / 1024})
    summary["latency_ms_pc"] = summary["setting"].map({"keras_fp32": lat32, "tflite_int8": lat8})
    summary.to_csv(os.path.join(C.RESULTS_DIR, "tinyml_results.csv"), index=False)
    pd.DataFrame(attack_rows).to_csv(os.path.join(C.RESULTS_DIR, "tinyml_per_attack.csv"), index=False)
    print(f"\nSaved: results/tinyml_results.csv, models/fl_ae_int8.tflite, models/fl_ae_int8.h")
    print("Note: latency is measured on this PC. On an ESP32 / Cortex-M it will be slower;"
          " measure it on the board for the paper.")


if __name__ == "__main__":
    main()
