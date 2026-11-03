# Federated TinyML Anomaly Detection for Zero-Trust Hospital IoT

**Author:** Saba Nawaz · MS Computer Science, UMT Lahore · ORCID [0009-0003-1686-935X](https://orcid.org/0009-0003-1686-935X)

Privacy-preserving intrusion detection for IoT devices that combines three ideas:

1. **Federated learning (FedAvg):** each device type trains an anomaly detector on its *own* traffic. Only model weights are shared, never raw data.
2. **TinyML:** the global model is quantized to **int8 (29 KB)** so it can run on a microcontroller (ESP32 / ARM Cortex-M) with TensorFlow Lite Micro.
3. **Zero Trust engine:** every device continuously proves it is not compromised. The on-device anomaly score updates a trust score that moves the device between **FULL**, **RESTRICTED** and **QUARANTINED** access.

This project extends my MS thesis (*lightweight Zero Trust architecture for IoT*), which used a simulated environment, to **real IoT botnet traffic** and to **on-device, privacy-preserving** detection.

---

## 1. System overview

```
 Hospital IoT devices (clients)                 Federated server
 ┌───────────────────────────┐   weights only   ┌──────────────────┐
 │ Doorbell     local data ──┼─────────────────▶│                  │
 │ Thermostat   local data ──┼─────────────────▶│  FedAvg: weighted│
 │ Baby monitor local data ──┼─────────────────▶│  average of      │
 │ Camera       local data ──┼─────────────────▶│  client weights  │
 └───────────────────────────┘◀─────────────────┴──────────────────┘
               │                global model
               ▼
 int8 TinyML autoencoder on each device (29 KB)
               │ anomaly score (reconstruction error) vs device's own threshold
               ▼
 Zero Trust engine: trust = 0.9·trust + 0.1·(0 if anomalous else 1)
               │
               ▼
 FULL (≥0.7)  →  RESTRICTED (0.4–0.7)  →  QUARANTINED (<0.4)
```

**Model:** dense autoencoder 115 → 58 → 29 → 14 → 29 → 58 → 115 (17,819 parameters), trained only on normal (benign) traffic. High reconstruction error = anomaly.

---

## 2. Dataset

**N-BaIoT** (Meidan et al., 2018): real traffic from commercial IoT devices infected with the Mirai and BASHLITE (Gafgyt) botnets. 115 statistical features per sample. 10 attack types.

Four devices are used as federated clients:

| Client | Device | Benign train / val / test | Attack test rows |
|---|---|---|---|
| Doorbell | Danmini Doorbell | 29,728 / 4,954 / 14,866 | 100,000 |
| Thermostat | Ecobee Thermostat | 7,867 / 1,311 / 3,935 | 100,000 |
| BabyMonitor | Philips B120N10 Baby Monitor | 105,144 / 17,524 / 52,572 | 100,000 |
| Camera | Provision PT-737E Security Camera | 37,292 / 6,215 / 18,647 | 100,000 |

The clients hold very different amounts of data (13× difference), a realistic **non-IID / unbalanced** federated setting. Attack rows: 10,000 sampled per attack type, test set only.

**Download used:** per-device archives from a GitHub release mirror (the official UCI zip could not be resumed on a slow connection). Integrity was verified: every file's row count and 115 columns match the original N-BaIoT numbers (e.g. Danmini benign = 49,548 rows). **Cite the original dataset** (see References).

---

## 3. How to run (Windows, Python 3.12)

### 3.1 Setup (once)
```
cd "C:\Users\aa\Documents\SAba Projects\fl-tinyml-zero-trust"
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 3.2 Get the data (once, ~975 MB download)
Run one line at a time inside the project folder:
```
curl -L -C - -o data\Danmini_Doorbell.tar.gz "https://github.com/manganjokean/computer-security-project/releases/download/nbaiot-data/Danmini_Doorbell.tar.gz"
curl -L -C - -o data\Ecobee_Thermostat.tar.gz "https://github.com/manganjokean/computer-security-project/releases/download/nbaiot-data/Ecobee_Thermostat.tar.gz"
curl -L -C - -o data\Philips_B120N10_Baby_Monitor.tar.gz "https://github.com/manganjokean/computer-security-project/releases/download/nbaiot-data/Philips_B120N10_Baby_Monitor.tar.gz"
curl -L -C - -o data\Provision_PT_737E_Security_Camera.tar.gz "https://github.com/manganjokean/computer-security-project/releases/download/nbaiot-data/Provision_PT_737E_Security_Camera.tar.gz"
tar -xzf data\Danmini_Doorbell.tar.gz -C data
tar -xzf data\Ecobee_Thermostat.tar.gz -C data
tar -xzf data\Philips_B120N10_Baby_Monitor.tar.gz -C data
tar -xzf data\Provision_PT_737E_Security_Camera.tar.gz -C data
```
If a download stops, run the same `curl` line again: it resumes. After extracting, the `.tar.gz` files can be deleted (`del data\*.tar.gz`).

### 3.3 Run the pipeline (in this order)
Every time you open a new Command Prompt, first run `venv\Scripts\activate`.

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `python 01_prepare_data.py` | ~2 min | Split, log-transform and scale each device's data locally |
| 2 | `python 02_baseline.py` | ~10–15 min | Local-only and centralized baselines |
| 3 | `python 03_federated.py` | ~4 min | Federated learning (FedAvg), 20 rounds |
| 4 | `python 04_tinyml.py` | ~2 min | int8 TFLite conversion, size/latency/accuracy, C header |
| 5 | `python 05_trust_engine.py` | ~2 min | Zero Trust engine simulation + trust timeline figure |
| 6 | `python 06_figures.py` | <1 min | Paper figures and summary table |

Repeat over seeds: `python run_seeds.py` (about 20–25 minutes per seed).
Optional: `python diagnose_floods.py` reproduces the log-transform experiment (Section 5.2).
Optional: `python 03_federated.py --dp-noise 0.5 --dp-clip 2.0` runs DP-FedAvg (see Section 6).

All settings (devices, model size, threshold, rounds) are in **`config.py`**.

---

## 4. Project files

| File | Purpose |
|---|---|
| `config.py` | All settings in one place |
| `common.py` | Data loading, autoencoder, threshold and metrics |
| `01_prepare_data.py` | Per-device preprocessing (no data shared between devices) |
| `02_baseline.py` | Baselines: local-only and centralized |
| `03_federated.py` | FedAvg (and optional DP-FedAvg) |
| `04_tinyml.py` | int8 quantization, model size, latency, accuracy, `fl_ae_int8.h` |
| `05_trust_engine.py` | Zero Trust engine driven by the int8 model |
| `06_figures.py` | Figures and summary table |
| `diagnose_floods.py` | Experiment showing why the log transform is used |
| `run_seeds.py` | Runs steps 1–5 for several seeds and reports mean ± std |
| `results/` | CSV results and `figures/` (PNG, 200 dpi) |
| `models/` | Saved Keras / TFLite models and the C header for firmware |

---

## 5. Results

### 5.0 Repeated over three seeds (main result)

Each seed changes the data split, the sampled attack rows and the model initialisation (`python run_seeds.py`, seeds 42, 1, 2). Mean ± standard deviation of the 4-device average:

| Setting | Raw data shared? | F1 | Recall | FPR | AUC |
|---|---|---|---|---|---|
| Local-only | No | 0.953 ± 0.016 | 0.917 ± 0.029 | 0.009 | 0.998 |
| Centralized | **Yes** | 0.934 ± 0.016 | 0.883 ± 0.029 | 0.010 | 0.997 |
| **Federated (FedAvg)** | **No** | **0.962 ± 0.016** | 0.933 ± 0.029 | 0.009 | 0.997 |
| Federated int8 (TinyML) | No | 0.980 ± 0.032 | 0.967 ± 0.058 | 0.010 | 0.998 |

Federated learning had the highest average F1 of the three training settings without sharing raw data. The int8 model's higher mean is not an improvement from quantization: it comes from borderline flood samples that fall on either side of the threshold (note its larger standard deviation). The correct reading is that int8 quantization did not reduce detection performance.

Zero Trust engine over the three seeds: 90–100% of attack instances led to quarantine, after about 9 samples on average, with **0.0% wrongly quarantined time** on normal traffic in every seed.

The detailed tables below are from a single run (seed 42).

Threshold: 99th percentile of each device's own benign validation errors (target false positive rate ≈ 1%), computed locally by each device.

### 5.1 Detection, average over 4 devices

| Setting | Raw data shared? | Accuracy | Precision | Recall | FPR | F1 | AUC |
|---|---|---|---|---|---|---|---|
| Local-only | No | 0.9177 | 0.9976 | 0.9001 | 0.0091 | 0.9435 | 0.9972 |
| Centralized | **Yes** | 0.8754 | 0.9973 | 0.8501 | 0.0098 | 0.9156 | 0.9976 |
| **Federated (FedAvg)** | **No** | **0.9234** | 0.9973 | 0.9000 | 0.0096 | **0.9433** | 0.9943 |
| **Federated int8 (TinyML)** | **No** | **0.9234** | 0.9972 | 0.9000 | 0.0099 | **0.9433** | 0.9945 |

**Finding 1:** federated learning matches local-only and outperforms centralized training on average F1, **without sharing any raw data**. Total communication: 11.4 MB of weights over 20 rounds.

**Finding 2:** the client with the least data (Thermostat, 7,867 training rows) only detects Gafgyt TCP/UDP floods in the **federated** setting (F1 0.9998 vs 0.8888 local and centralized): it benefits from what other clients learned.

### 5.2 Per attack type (federated)

8 of 10 attack types are detected at ≈100% on all devices. **Gafgyt TCP and UDP floods** are detected on Doorbell and Thermostat but **not** on Baby Monitor and Camera (<0.1%), in all three training settings. Video-streaming devices produce heavy, high-rate normal traffic that resembles a flood when each sample is judged alone.

A **signed log transform** (`sign(x)·log(1+|x|)`) before scaling is used because, in `diagnose_floods.py`, it raised flood detection on some devices from ~0% to 100% with no increase in false positives. Flood detection is sensitive to training length and setting, so it should be confirmed over several seeds (Section 6).

### 5.3 TinyML

| Model | Size | Latency per sample (PC) | Peak activations | F1 (avg) |
|---|---|---|---|---|
| Keras float32 | 69.6 KB | – | – | 0.9433 |
| TFLite float32 | 73.0 KB | 0.017 ms | 692 B | – |
| **TFLite int8** | **29.3 KB** | **0.012 ms** | **173 B** | **0.9433** |

**Finding 3:** int8 quantization makes the model 2.5× smaller with **no measurable loss** (F1 change ≤ 0.0001). 29 KB fits easily on an ESP32 (520 KB SRAM). Latency was measured on a PC; it must be measured on a real board for publication.

### 5.4 Zero Trust engine (int8 model, decay 0.9)

Stream per device and attack: 300 normal → 100 attack → 300 normal samples.

| | Doorbell | Thermostat | BabyMonitor | Camera |
|---|---|---|---|---|
| Samples until QUARANTINE (8 detected attack types) | 8–9 | 9 | 9 | 8–9 |
| Gafgyt TCP / UDP floods | 9 | 9 | never | never |
| Time wrongly restricted during normal traffic | 0.000% | 0.000% | 0.000% | 0.000% |

**Finding 4:** although the detector alone flags ~1% of normal samples, the trust engine wrongly restricted a device **0 times in 90,020 normal samples**, while quarantining a compromised device within **8–9 samples**. Isolated false alarms are absorbed; sustained anomalies are not.

Figures (in `results/figures/`): `f1_comparison.png`, `fl_convergence.png`, `per_attack_heatmap.png`, `trust_timeline.png`.

---

## 6. Limitations and next steps

1. **More seeds:** results above use three seeds (`run_seeds.py`). Five or more seeds would give tighter confidence intervals; flood detection in particular varies between runs.
2. **Flood blind spot on streaming devices:** judge a *window* of consecutive samples (as the original N-BaIoT paper did) instead of single samples, or add rate-based features.
3. **Differential privacy:** DP-FedAvg is implemented (`--dp-noise`), but with only 4 clients the added noise is far larger than the real updates (noise is divided by the number of clients). It needs many more clients (e.g. split each device's data into 10–20 simulated clients) before it gives useful results.
4. **Real hardware:** flash `models/fl_ae_int8.h` to an ESP32 with TensorFlow Lite Micro and measure real latency, RAM and energy. The on-device preprocessing is cheap: signed log + per-feature scaling (`scaler_mean`, `scaler_scale` in `data/processed/*.npz`).
5. **More devices / cross-dataset:** add the other 5 N-BaIoT devices, then test on CICIoT2023 (links to my planned Publication 2 on cross-dataset generalization).
6. **Framework:** FedAvg is implemented directly (algorithm of McMahan et al., 2017). Porting to Flower is possible but its simulation engine is unreliable on native Windows.

---

## 7. References

- Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, A. Shabtai, D. Breitenbacher, Y. Elovici, "N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders," *IEEE Pervasive Computing*, 17(3), 2018. Dataset: UCI Machine Learning Repository, doi:10.24432/C5RC8J (CC BY 4.0).
- B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," *AISTATS*, 2017 (FedAvg).
- H. B. McMahan, D. Ramage, K. Talwar, L. Zhang, "Learning differentially private recurrent language models," *ICLR*, 2018 (DP-FedAvg).
- R. David et al., "TensorFlow Lite Micro: Embedded machine learning on TinyML systems," *MLSys*, 2021.
