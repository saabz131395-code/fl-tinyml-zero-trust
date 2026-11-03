"""
Shared settings for the FL-TinyML Zero Trust project.
Every script imports from here, so change a setting once and all scripts follow.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")                 # raw N-BaIoT CSVs
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")       # prepared .npz files
RESULTS_DIR = os.path.join(BASE_DIR, "results")           # tables and graphs
MODELS_DIR = os.path.join(BASE_DIR, "models")             # saved models

# ---------------------------------------------------------------------------
# Devices = federated clients (one hospital IoT device type each)
# ---------------------------------------------------------------------------
DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
]

# Short names for tables and graphs
SHORT_NAMES = {
    "Danmini_Doorbell": "Doorbell",
    "Ecobee_Thermostat": "Thermostat",
    "Philips_B120N10_Baby_Monitor": "BabyMonitor",
    "Provision_PT_737E_Security_Camera": "Camera",
}

ATTACK_FILES = {
    "gafgyt": ["combo", "junk", "scan", "tcp", "udp"],
    "mirai": ["ack", "scan", "syn", "udp", "udpplain"],
}

# ---------------------------------------------------------------------------
# Data split (follows the original N-BaIoT paper: train only on benign data)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.60          # benign rows used to train the autoencoder
VAL_FRAC = 0.10            # benign rows used to pick the anomaly threshold
                           # remaining 30% of benign rows go to the test set
LOG_TRANSFORM = True        # apply sign(x)*log(1+|x|) before scaling; needed to detect
                            # Gafgyt TCP/UDP floods (see diagnose_floods.py)
MAX_ROWS_PER_ATTACK = 10000  # attack rows sampled per attack file (keeps RAM low)
RANDOM_SEED = int(os.environ.get("SEED", 42))   # run_seeds.py sets SEED

# ---------------------------------------------------------------------------
# Autoencoder (small on purpose, so it fits on a microcontroller later)
# 115 -> 58 -> 29 -> 14 -> 29 -> 58 -> 115
# ---------------------------------------------------------------------------
N_FEATURES = 115
HIDDEN_LAYERS = [58, 29, 14]
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50                # centralized / local training (early stopping may end sooner)
EARLY_STOP_PATIENCE = 5

# Threshold rule: this percentile of reconstruction errors on benign validation data.
# 99 means about 1% of normal traffic may be flagged (target false positive rate ~1%).
THRESHOLD_PERCENTILE = 99.0

for _d in (PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Federated learning (FedAvg)
# ---------------------------------------------------------------------------
FL_ROUNDS = 20             # communication rounds between server and clients
LOCAL_EPOCHS = 2           # epochs each client trains per round
