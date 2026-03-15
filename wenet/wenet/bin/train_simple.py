#!/usr/bin/env python
"""
Simple training script for single GPU/CPU without distributed training
"""

import os
import sys
import argparse
import logging
import yaml
import torch

# Set environment variables before importing wenet
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Add wenet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import init_optimizer_and_scheduler, init_summarywriter
from wenet.utils.executor import Executor
from wenet.utils.config import override_config
import wenet.utils.train_utils as train_utils


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
    args = parser.parse_args()

    # Set logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # Init tokenizer
    tokenizer = init_tokenizer(configs)

    # Mock args for train_utils
    class Args:
        train_engine = 'torch_ddp'
        device = args.device
        dist_backend = 'gloo'
        use_amp = False
        jit = False
        print_model = False

    args_train = Args()

    # Get dataset - simplified (would need proper dataloader setup)
    # For now, we'll skip actual training loop setup

    # Init model
    model, configs = init_model(args_train, configs)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args_train, configs, model)

    logging.info(f"Optimizer: {type(optimizer)}")
    logging.info("Model initialized successfully!")
    logging.info("This is a simplified training script.")
    logging.info("For full training, please use Docker or WSL2.")


if __name__ == '__main__':
    main()
