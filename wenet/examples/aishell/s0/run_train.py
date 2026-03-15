#!/usr/bin/env python
import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

# Change to the example directory
os.chdir(script_dir)

# Run training
cmd = [
    sys.executable,
    os.path.join(repo_root, 'wenet', 'bin', 'train.py'),
    '--train_engine', 'torch_ddp',
    '--device', 'cpu',
    '--config', 'conf/train_conformer.yaml',
    '--data_type', 'raw',
    '--train_data', 'data/train/data.list',
    '--cv_data', 'data/dev/data.list',
    '--model_dir', 'exp/pogo_test',
    '--num_workers', '0',
    '--ddp.dist_backend', 'gloo'
]

subprocess.run(cmd)
