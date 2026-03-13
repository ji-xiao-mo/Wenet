#!/bin/bash
# ROOT Optimizer Training Script for Aishell

# Set Python environment
export PYTHONPATH=/e/vscode/wenet
export PATH=/e/vscode/wenet/Anacondaenvs/wenet:/e/vscode/wenet/Anacondaenvs/wenet/Scripts:$PATH

# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cd /e/vscode/wenet/examples/aishell/s0 || exit 1

# Training config
config=conf/train_conformer_root.yaml
train_data=data/train/data.list
cv_data=data/dev/data.list
model_dir=exp/conformer_root

echo "Starting training with ROOT optimizer..."
echo "Config: $config"
echo "Train data: $train_data"
echo "CV data: $cv_data"
echo "Model dir: $model_dir"

python wenet/bin/train.py \
    --config "$config" \
    --train_data "$train_data" \
    --cv_data "$cv_data" \
    --model_dir "$model_dir" \
    --train_engine torch_ddp \
    --device cuda \
    --num_workers 2 \
    --prefetch 2
