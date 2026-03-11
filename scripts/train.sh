#!/bin/bash
# ==========================================
# Automated Training Launcher for CADT
# ==========================================

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve the absolute path of the project root
PROJECT_ROOT=$(cd "$(dirname "$0")/.."; pwd)

echo "======================================================"
echo "[INFO] Starting Training Pipeline for Drowning Detection"
echo "[INFO] Project Root: ${PROJECT_ROOT}"
echo "======================================================"

# Navigate to project root to ensure relative paths in Python resolve correctly
cd ${PROJECT_ROOT}

# Step 1: Check if the dataset artifact exists, generate if not
DATA_FILE="${PROJECT_ROOT}/data/train_dataset.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "[WARNING] Training dataset not found at ${DATA_FILE}."
    echo "[INFO] Initiating data generation protocol..."
    python scripts/generate_csv.py
else
    echo "[INFO] Dataset located. Skipping generation."
fi

# Step 2: Extract hyperparameters from YAML (Simple grep parser for bash)
# Note: In Python, pyyaml will handle the full parsing.
EPOCHS=$(grep "epochs:" configs/train_config.yaml | awk '{print $2}')
BATCH_SIZE=$(grep "batch_size:" configs/train_config.yaml | awk '{print $2}')
DEVICE=$(grep "device_target:" configs/train_config.yaml | awk -F '"' '{print $2}')

echo "[INFO] Using configuration -> Device: ${DEVICE}, Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"

# Step 3: Launch the MindSpore training script
echo "[INFO] Launching train.py..."
python train.py \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE}

echo "======================================================"
echo "[SUCCESS] Training Pipeline Terminated Successfully."
echo "======================================================"