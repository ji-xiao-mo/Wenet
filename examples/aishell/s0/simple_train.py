#!/usr/bin/env python3
"""Simple training script that bypasses torchrun for Windows."""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist

# Add wenet to path
sys.path.insert(0, 'E:/vscode/wenet')


def main():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--train_engine', type=str, default='torch_ddp',
                        choices=['torch_ddp', 'torch_fsdp', 'deepspeed'])
    parser.add_argument('--config', type=str, required=True,
                        help='config file')
    parser.add_argument('--data_type', type=str, default='raw',
                        choices=['raw', 'shard'])
    parser.add_argument('--train_data', type=str, required=True,
                        help='train data file')
    parser.add_argument('--cv_data', type=str, required=True,
                        help='cv data file')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='save model dir')
    parser.add_argument('--tensorboard_dir', type=str,
                        help='tensorboard log dir')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint model')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers for dataloader')
    parser.add_argument('--prefetch', type=int, default=0,
                        help='prefetch number')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='data type for training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use for training')
    parser.add_argument('--dist_backend', type=str, default='gloo',
                        help='distributed backend')
    parser.add_argument('--pin_memory', action='store_true',
                        help='pin_memory for dataloader')
    parser.add_argument('--enable_profiling', action='store_true',
                        help='enable profiling for model')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='log interval')
    parser.add_argument('--reset_data', action='store_true',
                        help='reset data iterator')
    parser.add_argument('--multi_task', action='store_true',
                        help='use multi task loss')
    parser.add_argument('--use_rtensor', action='store_true',
                        help='use ragged tensor for dataset')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize distributed using tcp:// init method to avoid env://
    logging.info('Initializing distributed training (single GPU mode)...')

    # Use TCP init method with GlooStore (no libuv needed)
    init_method = "tcp://localhost:29500"
    world_size = 1
    rank = 0

    # Create a TCPStore for single machine (workaround for libuv issue)
    # Actually, we need to use GlooStore directly or FileStore
    # For single process, we can use a simpler approach

    # Use FileStore for single-process training (no network needed)
    store = dist.FileStore("E:/vscode/wenet/examples/aishell/s0/filestore")
    dist.init_process_group(
        backend='gloo',
        init_method='file://E:/vscode/wenet/examples/aishell/s0/filestore',
        world_size=1,
        rank=0
    )

    # Rest of the training code would go here...
    logging.info('Distributed initialized successfully!')
    logging.info(f'Rank: {dist.get_rank()}, World size: {dist.get_world_size()}')

    # For now, just verify it works
    logging.info('Training script initialized successfully!')
    dist.destroy_process_group()
    logging.info('Done!')


if __name__ == '__main__':
    main()
