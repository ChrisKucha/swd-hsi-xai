"""
config.py — All configuration constants for the SWD blueberry HSI pipeline.

Edit this file before running. Nothing else needs to change.
"""

import os
import random
import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Binary classification ─────────────────────────────────────────────────────
CLASS_NAMES   = ["Infested", "Healthy"]   # index 0 = Infested, 1 = Healthy
NUM_CLASSES   = 2
CLASS_COLORS  = {0: "red", 1: "limegreen"}

# Ripeness labels (used in multi-task mode)
RIPENESS_NAMES  = ["Ripe", "Midripe", "Unripe"]
NUM_RIPENESS    = 3

# ── Board layout ──────────────────────────────────────────────────────────────
BERRIES_PER_BOARD = 36
BERRY_GRID_ROWS   = 6
BERRY_GRID_COLS   = 6

# ── Data paths (second approach — board-identity-safe) ────────────────────────
# Structure: {class}_npy_files_combine_{sensor} / Day{N} / {Ripeness} / *.npy
#
# Raw cube data and generated shards live on the mounted New Volume drive by
# default. Override SWD_DATA_ROOT, SWD_*_HEALTHY, or SWD_SHARD_DIR if needed.
DATA_ROOT = os.environ.get("SWD_DATA_ROOT", "/media/kuchalab/New Volume")

SMB_NIR_INFECTED  = os.environ.get(
    "SWD_NIR_INFECTED",
    os.path.join(DATA_ROOT, "Infected_npy_files_combine_nir"),
)
SMB_NIR_HEALTHY   = os.environ.get(
    "SWD_NIR_HEALTHY",
    os.path.join(DATA_ROOT, "Healthy_npy_files_combine_nir"),
)
SMB_VNIR_INFECTED = os.environ.get(
    "SWD_VNIR_INFECTED",
    os.path.join(DATA_ROOT, "Infected_npy_files_combine_vnir"),
)
SMB_VNIR_HEALTHY  = os.environ.get(
    "SWD_VNIR_HEALTHY",
    os.path.join(DATA_ROOT, "Healthy_npy_files_combine_vnir"),
)

_PATHS = {
    "nir":  {"Infested": SMB_NIR_INFECTED,  "Healthy": SMB_NIR_HEALTHY},
    "vnir": {"Infested": SMB_VNIR_INFECTED, "Healthy": SMB_VNIR_HEALTHY},
}
PATHS = _PATHS

SHARD_DIR = os.environ.get(
    "SWD_SHARD_DIR",
    os.path.join(DATA_ROOT, "swd_detection_shards"),
)

RIPENESS_STAGES = ["Ripe", "Midripe", "Unripe"]
DAYS            = list(range(1, 7))   # Day 1 … Day 6

# ── Cropping ──────────────────────────────────────────────────────────────────
NIR_TRIM_TOP,  NIR_TRIM_BOTTOM,  NIR_TRIM_LEFT,  NIR_TRIM_RIGHT  = 0, 60, 40,  100
VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM, VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT = 0, 60, 165, 100

# ── VNIR horizontal flip ──────────────────────────────────────────────────────
# NIR is the spatial reference. VNIR must be flipped left-right after cropping
# so that cell (row, col) refers to the same physical berry in both sensors.
VNIR_FLIP_HORIZONTAL = True

# ── Exact wavelength centers (nm) for axis labels, attribution, and band select ─
VNIR_WAVELENGTHS = [
    397.66, 400.28, 402.90, 405.52, 408.13, 410.75, 413.37, 416.00,
    418.62, 421.24, 423.86, 426.49, 429.12, 431.74, 434.37, 437.00,
    439.63, 442.26, 444.89, 447.52, 450.16, 452.79, 455.43, 458.06,
    460.70, 463.34, 465.98, 468.62, 471.26, 473.90, 476.54, 479.18,
    481.83, 484.47, 487.12, 489.77, 492.42, 495.07, 497.72, 500.37,
    503.02, 505.67, 508.32, 510.98, 513.63, 516.29, 518.95, 521.61,
    524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91,
    545.57, 548.24, 550.91, 553.58, 556.25, 558.92, 561.59, 564.26,
    566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583.00, 585.68,
    588.36, 591.04, 593.73, 596.41, 599.10, 601.78, 604.47, 607.16,
    609.85, 612.53, 615.23, 617.92, 620.61, 623.30, 626.00, 628.69,
    631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29,
    652.99, 655.69, 658.40, 661.10, 663.81, 666.52, 669.23, 671.94,
    674.65, 677.36, 680.07, 682.79, 685.50, 688.22, 690.93, 693.65,
    696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.70, 715.42,
    718.15, 720.87, 723.60, 726.33, 729.06, 731.79, 734.52, 737.25,
    739.98, 742.72, 745.45, 748.19, 750.93, 753.66, 756.40, 759.14,
    761.88, 764.62, 767.36, 770.11, 772.85, 775.60, 778.34, 781.09,
    783.84, 786.58, 789.33, 792.08, 794.84, 797.59, 800.34, 803.10,
    805.85, 808.61, 811.36, 814.12, 816.88, 819.64, 822.40, 825.16,
    827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52, 847.29,
    850.06, 852.83, 855.60, 858.37, 861.14, 863.92, 866.69, 869.47,
    872.25, 875.03, 877.80, 880.58, 883.37, 886.15, 888.93, 891.71,
    894.50, 897.28, 900.07, 902.86, 905.64, 908.43, 911.22, 914.02,
    916.81, 919.60, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38,
    939.18, 941.98, 944.78, 947.58, 950.38, 953.19, 955.99, 958.80,
    961.60, 964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27,
    984.09, 986.90, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81,
]

NIR_WAVELENGTHS = [
    935.61, 939.06, 942.52, 945.98, 949.43, 952.89, 956.35, 959.81,
    963.27, 966.73, 970.19, 973.65, 977.11, 980.58, 984.04, 987.51,
    990.97, 994.43, 997.90, 1001.37, 1004.83, 1008.30, 1011.77, 1015.24,
    1018.71, 1022.18, 1025.65, 1029.12, 1032.59, 1036.06, 1039.53, 1043.00,
    1046.48, 1049.95, 1053.43, 1056.90, 1060.38, 1063.85, 1067.33, 1070.81,
    1074.29, 1077.76, 1081.24, 1084.72, 1088.20, 1091.68, 1095.17, 1098.65,
    1102.13, 1105.61, 1109.10, 1112.58, 1116.07, 1119.55, 1123.04, 1126.52,
    1130.01, 1133.50, 1136.99, 1140.47, 1143.96, 1147.45, 1150.94, 1154.43,
    1157.93, 1161.42, 1164.91, 1168.40, 1171.90, 1175.39, 1178.89, 1182.38,
    1185.88, 1189.37, 1192.87, 1196.37, 1199.87, 1203.37, 1206.87, 1210.37,
    1213.87, 1217.37, 1220.87, 1224.37, 1227.87, 1231.38, 1234.88, 1238.39,
    1241.89, 1245.40, 1248.90, 1252.41, 1255.92, 1259.42, 1262.93, 1266.44,
    1269.95, 1273.46, 1276.97, 1280.48, 1283.99, 1287.51, 1291.02, 1294.53,
    1298.05, 1301.56, 1305.08, 1308.59, 1312.11, 1315.62, 1319.14, 1322.66,
    1326.18, 1329.70, 1333.22, 1336.74, 1340.26, 1343.78, 1347.30, 1350.82,
    1354.35, 1357.87, 1361.39, 1364.92, 1368.44, 1371.97, 1375.50, 1379.02,
    1382.55, 1386.08, 1389.61, 1393.14, 1396.67, 1400.20, 1403.73, 1407.26,
    1410.79, 1414.32, 1417.86, 1421.39, 1424.92, 1428.46, 1431.99, 1435.53,
    1439.07, 1442.60, 1446.14, 1449.68, 1453.22, 1456.76, 1460.30, 1463.84,
    1467.38, 1470.92, 1474.46, 1478.01, 1481.55, 1485.09, 1488.64, 1492.18,
    1495.73, 1499.27, 1502.82, 1506.37, 1509.91, 1513.46, 1517.01, 1520.56,
    1524.11, 1527.66, 1531.21, 1534.76, 1538.32, 1541.87, 1545.42, 1548.98,
    1552.53, 1556.09, 1559.64, 1563.20, 1566.75, 1570.31, 1573.87, 1577.43,
    1580.99, 1584.55, 1588.11, 1591.67, 1595.23, 1598.79, 1602.35, 1605.92,
    1609.48, 1613.04, 1616.61, 1620.17, 1623.74, 1627.31, 1630.87, 1634.44,
    1638.01, 1641.58, 1645.15, 1648.71, 1652.29, 1655.86, 1659.43, 1663.00,
    1666.57, 1670.14, 1673.72, 1677.29, 1680.87, 1684.44, 1688.02, 1691.59,
    1695.17, 1698.75, 1702.33, 1705.91, 1709.49, 1713.07, 1716.65, 1720.23,
]

WAVELENGTHS = {
    "nir": NIR_WAVELENGTHS,
    "vnir": VNIR_WAVELENGTHS,
}

# ── Band selection (None = use all bands) ─────────────────────────────────────
NIR_SELECTED_BANDS  = None
VNIR_SELECTED_BANDS = None

# ── Cell size (pixels) — None = infer from first board at runtime ─────────────
# If set, all cells are resized to (CELL_H, CELL_W) before feeding the model.
# Recommended: leave as None during exploration; fix once cube dimensions confirmed.
CELL_H = None
CELL_W = None

# ── Train / val / test split ──────────────────────────────────────────────────
# Split is at BOARD identity level: all 6 days of one board go to the same split.
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20   # remainder

# ── Training mode ─────────────────────────────────────────────────────────────
# "stage_agnostic" : all ripeness stages pooled, binary classifier
# "per_stage"      : one model per ripeness stage (Ripe / Midripe / Unripe)
# "multi_task"     : joint prediction of class + ripeness
TRAINING_MODE = "stage_agnostic"

# ── Sensor mode ───────────────────────────────────────────────────────────────
# "nir" | "vnir"
# RUN_ALL_SENSORS = True runs both sequentially.
SENSOR_MODE    = "nir"
RUN_ALL_SENSORS = True

# ── Model selection ───────────────────────────────────────────────────────────
# "cnn3d" | "cnn3d_transformer"
# RUN_ALL_MODELS = True runs both sequentially.
MODEL_NAME    = "cnn3d"
RUN_ALL_MODELS = True

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE     = 16     #NIR worked best at 32
EVAL_BATCH_SIZE = 2     #NIR worked best at 4
NUM_WORKERS    = 4      #NIR worked best at 8
NUM_EPOCHS     = 250
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.5
LABEL_SMOOTHING = 0.05   # reduces overconfidence on small dataset
MAX_GRAD_NORM  = 1.0     # gradient clipping — prevents exploding gradients

# Early stopping
EARLY_STOP_PATIENCE = 30   # epochs with no val-loss improvement
LR_PATIENCE         = 7    # epochs before ReduceLROnPlateau fires

# Mixed precision training (requires CUDA Ampere+ for full benefit)
USE_AMP = True

# ── 3D-CNN architecture ───────────────────────────────────────────────────────
CNN3D_CHANNELS      = [32, 64, 128]          # filters per conv block
CNN3D_SPECTRAL_KERN = 7                      # spectral kernel size (first layer)
CNN3D_SPATIAL_KERN  = 3                      # spatial kernel size
CNN3D_FC_DIM        = 256

# ── 3D-CNN + Transformer ──────────────────────────────────────────────────────
TRANSFORMER_D_MODEL  = 256
TRANSFORMER_N_HEADS  = 8
TRANSFORMER_N_LAYERS = 4
TRANSFORMER_DFF      = 512
TRANSFORMER_DROPOUT  = 0.1

# ── Multispectral band-sweep ──────────────────────────────────────────────────
# Band counts tested by run_multispectral.py.
# For each K the top-K bands (by IG attribution) are selected and a new model
# is trained from scratch, simulating a K-band multispectral camera.

# MS_BAND_COUNTS = [5, 10, 15, 20, 25, 30]
MS_BAND_COUNTS = [30]

# Band-selection strategies tested by the automatic multispectral cascade.
# Use ["averaged"] for both-class attribution only, or ["infested", "averaged"]
# when you want to compare infested-only selection against averaged selection.
# MS_STRATEGIES = ["infested", "averaged"]
MS_STRATEGIES = ["averaged"]

# ── Output ────────────────────────────────────────────────────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()

OUTPUT_DIR = os.environ.get(
    "SWD_OUTPUT_DIR",
    os.path.join(_SCRIPT_DIR, "outputs")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
