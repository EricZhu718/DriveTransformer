"""
Test script to load dataloader without model
"""
import argparse
import torch
import os
import time
from mmcv.datasets import build_dataset
from mmcv.utils import Config, get_root_logger, mkdir_or_exist, set_random_seed, get_dist_info, init_dist
from adzoo.drivetransformer.mmdet3d_plugin.datasets.builder import build_dataloader
from datetime import timedelta
import cv2
cv2.setNumThreads(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Test dataloader loading')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', type=str, default='./work_dirs/dataloader_test', help='store work dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=5, help='number of samples to load')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Import plugin modules
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing plugin: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f"Importing plugin: {_module_path}")
                plg_lib = importlib.import_module(_module_path)

    # Set work_dir
    cfg.work_dir = args.work_dir
    cfg.gpu_ids = range(1)

    # Init distributed env if needed
    if args.launcher == 'none':
        distributed = False
    elif args.launcher == 'pytorch':
        distributed = True
        init_dist(args.launcher, timeout=timedelta(minutes=30), **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        distributed = False
    
    # Create work_dir
    mkdir_or_exist(os.path.abspath(cfg.work_dir))
    
    # Set random seed
    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=False)
    
    # Setup logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}_dataloader_test.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    logger.info('='*80)
    logger.info('Testing Dataloader Loading (Without Model)')
    logger.info('='*80)
    logger.info(f'Config: {args.config}')
    logger.info(f'Distributed: {distributed}')
    logger.info(f'Number of samples to load: {args.num_samples}')
    
    # Build dataset
    logger.info('\n' + '-'*80)
    logger.info('Building dataset...')
    datasets = [build_dataset(cfg.data.train)]
    logger.info(f'Dataset type: {type(datasets[0]).__name__}')
    logger.info(f'Dataset length: {len(datasets[0])}')
    
    # Build dataloader
    logger.info('\n' + '-'*80)
    logger.info('Building dataloader...')
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        runner_type=cfg.runner,
    ) for ds in datasets]
    
    logger.info(f'Dataloader created successfully!')
    logger.info(f'Batch size: {cfg.data.samples_per_gpu}')
    logger.info(f'Num workers: {cfg.data.workers_per_gpu}')
    
    # Try to load some samples
    logger.info('\n' + '-'*80)
    logger.info(f'Loading {args.num_samples} sample(s) from dataloader...')
    
    try:
        data_loader = data_loaders[0]
        for i, data_batch in enumerate(data_loader):
            if i >= args.num_samples:
                break
            
            logger.info(f'\n--- Sample {i+1} ---')
            logger.info(f'Data batch keys: {data_batch.keys()}')
            
            # Print shape information for each key
            for key, value in data_batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f'  {key}: Tensor, shape={value.shape}, dtype={value.dtype}')
                elif isinstance(value, list):
                    logger.info(f'  {key}: List, length={len(value)}')
                    if len(value) > 0 and isinstance(value[0], torch.Tensor):
                        logger.info(f'    First element: Tensor, shape={value[0].shape}, dtype={value[0].dtype}')
                elif isinstance(value, dict):
                    logger.info(f'  {key}: Dict with keys={list(value.keys())}')
                else:
                    logger.info(f'  {key}: {type(value).__name__}')
        
        logger.info('\n' + '='*80)
        logger.info('SUCCESS! Dataloader works without model.')
        logger.info('='*80)
        
    except Exception as e:
        logger.error('\n' + '='*80)
        logger.error('ERROR while loading data!')
        logger.error('='*80)
        logger.error(f'Error type: {type(e).__name__}')
        logger.error(f'Error message: {str(e)}')
        import traceback
        logger.error(f'Traceback:\n{traceback.format_exc()}')
        raise


if __name__ == '__main__':
    main()
