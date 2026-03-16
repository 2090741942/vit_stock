#!/usr/bin/env bash

set -e

PYTHON_BIN=${1:-python}

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# 普通包
packages=(
  numpy
  pandas
  tqdm
  transformers
)

# PyTorch CUDA index
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "---- Checking normal packages ----"

for pkg in "${packages[@]}"; do
  if "$PYTHON_BIN" -m pip show "$pkg" > /dev/null 2>&1; then
    echo "[OK] $pkg already installed"
  else
    echo "[INSTALL] $pkg"
    "$PYTHON_BIN" -m pip install "$pkg"
  fi
done


echo "---- Checking PyTorch ----"

if "$PYTHON_BIN" -c "import torch" > /dev/null 2>&1; then
  echo "[OK] torch already installed"
else
  echo "[INSTALL] torch (CUDA 12.8)"
  "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url $TORCH_INDEX
fi


echo "All done."

echo "---- Testing Torch CUDA ----"

"$PYTHON_BIN" - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF