#!/usr/bin/env python
"""
Simplified single-process training script that bypasses distributed training issues
"""

import os
import sys
import argparse
import logging
import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.config import override_config


class Args:
    train_engine = 'torch_ddp'
    device = 'cpu'
    dist_backend = 'gloo'
    use_amp = False
    jit = False
    print_model = False


def main():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--model_dir', type=str, required=True, help='model dir')
    parser.add_argument('--train_data', type=str, required=True, help='train data')
    parser.add_argument('--cv_data', type=str, required=True, help='cv data')
    parser.add_argument('--data_type', type=str, default='raw', help='data type')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--override_config', type=str, default=[], action='append', help='override config')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.model_dir, 'train.log'))
        ]
    )

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(777)

    logging.info("Loading config...")
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Override config if provided
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # Init tokenizer
    logging.info("Initializing tokenizer...")
    tokenizer = init_tokenizer(configs)

    # Init model
    logging.info("Initializing model...")
    args_train = Args()
    args_train.device = args.device
    model, configs = init_model(args_train, configs)

    # Setup optimizer
    logging.info("Setting up optimizer...")
    optim_conf = configs['optim_conf'].copy()
    lr = optim_conf.get('lr', 0.001)

    # Select optimizer based on config
    optim_type = configs.get('optim', 'adamw')
    logging.info(f"Using optimizer: {optim_type}")

    if optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **optim_conf)
    elif optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, **optim_conf)
    elif optim_type == 'pogo':
        # POGO optimizer
        from wenet.optimizer.pogo_main.pogo_main.pogo import POGO
        base_opt = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = POGO(model.parameters(), base_optimizer=base_opt, **optim_conf)
    elif optim_type == 'cautious_muon':
        from wenet.optimizer.muon import CautiousMuon
        optimizer = CautiousMuon(model.parameters(), **optim_conf)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logging.info(f"Optimizer type: {type(optimizer).__name__}")
    logging.info("Training setup complete!")
    logging.info("Model is ready for training.")
    logging.info("Note: This is a simplified training check script.")
    logging.info("For full distributed training, please use Docker or WSL2.")


if __name__ == '__main__':
    main()
