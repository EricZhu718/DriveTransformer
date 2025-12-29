import argparse
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import copy
import os
import time
import warnings
from os import path as osp
from mmcv import __version__ as mmcv_version
from mmcv.datasets import build_dataset
from mmcv.models import build_model
from mmcv.utils import collect_env, get_root_logger, mkdir_or_exist, set_random_seed, get_dist_info, init_dist, Config, DictAction, TORCH_VERSION, digit_version
from adzoo.drivetransformer.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.optims import build_optimizer
from torch.nn.parallel import DataParallel, DistributedDataParallel
from mmcv.core.evaluation.eval_hooks import CustomDistEvalHook
from mmcv.core import EvalHook
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner, Fp16OptimizerHook, OptimizerHook, build_runner)
from datetime import datetime, timedelta
import cv2
cv2.setNumThreads(1)


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv

def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


# Added by Claude Code
def load_checkpoint_partial(checkpoint_path: str, model: nn.Module, logger=None):
    """Load checkpoint with partial key matching.

    This function loads weights from a checkpoint, only loading keys that exist
    in both the checkpoint and the model. Useful when:
    - Loading checkpoint without diffusion_head into model with diffusion_head
    - Loading checkpoint with diffusion_head into model without diffusion_head

    Args:
        checkpoint_path: Path to checkpoint file
        model: The model to load weights into
        logger: Optional logger for printing information

    Returns:
        None (loads weights in-place into model)
    """
    if logger:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = checkpoint["state_dict"]

    # Get model state dict
    model_state = model.state_dict()

    # Find matching keys
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    matching_keys = checkpoint_keys & model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    missing_in_model = checkpoint_keys - model_keys

    # Load matching weights
    loaded_state = {k: checkpoint_state[k] for k in matching_keys}
    model.load_state_dict(loaded_state, strict=False)

    # Log results
    if logger:
        logger.info(f"Successfully loaded {len(matching_keys)}/{len(checkpoint_keys)} keys from checkpoint")

        # Group missing keys by whether they're diffusion-related
        diffusion_missing_in_ckpt = [k for k in missing_in_checkpoint if 'diffusion' in k.lower()]
        diffusion_missing_in_model = [k for k in missing_in_model if 'diffusion' in k.lower()]
        other_missing_in_ckpt = [k for k in missing_in_checkpoint if 'diffusion' not in k.lower()]
        other_missing_in_model = [k for k in missing_in_model if 'diffusion' not in k.lower()]

        if diffusion_missing_in_ckpt:
            logger.info(f"\nModel has {len(diffusion_missing_in_ckpt)} diffusion keys not in checkpoint (will use initialized weights):")
            for k in sorted(diffusion_missing_in_ckpt)[:5]:
                logger.info(f"  - {k}")
            if len(diffusion_missing_in_ckpt) > 5:
                logger.info(f"  ... and {len(diffusion_missing_in_ckpt) - 5} more")

        if diffusion_missing_in_model:
            logger.info(f"\nCheckpoint has {len(diffusion_missing_in_model)} diffusion keys not in model (skipped):")
            for k in sorted(diffusion_missing_in_model)[:5]:
                logger.info(f"  - {k}")
            if len(diffusion_missing_in_model) > 5:
                logger.info(f"  ... and {len(diffusion_missing_in_model) - 5} more")

        if other_missing_in_ckpt:
            logger.warning(f"\nWARNING: Model has {len(other_missing_in_ckpt)} non-diffusion keys not in checkpoint:")
            for k in sorted(other_missing_in_ckpt)[:10]:
                logger.warning(f"  - {k}")
            if len(other_missing_in_ckpt) > 10:
                logger.warning(f"  ... and {len(other_missing_in_ckpt) - 10} more")

        if other_missing_in_model:
            logger.warning(f"\nWARNING: Checkpoint has {len(other_missing_in_model)} non-diffusion keys not in model:")
            for k in sorted(other_missing_in_model)[:10]:
                logger.warning(f"  - {k}")
            if len(other_missing_in_model) > 10:
                logger.warning(f"  ... and {len(other_missing_in_model) - 10} more")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--fuse_conv_bn',
        action='store_true',
        help='whether to use fuse conv bn')
    parser.add_argument(
        '--compile_before',
        action='store_true',
        help='whether to use torch.compile')
    parser.add_argument(
        '--compile_after',
        action='store_true',
        help='whether to use torch.compile')
    parser.add_argument(
        '--remove_orig_mod',
        action='store_true',
        help='Whether to remove orig_mod in model_dict')
    parser.add_argument(
        '--partial-load',
        action='store_true',
        help='Load checkpoint with partial key matching (useful when checkpoint and model have different modules like diffusion_head)')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--work-dir', type=str, default=None, help='store work dir')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    # if cfg.get('custom_imports', None):
    #     from mmcv.utils import import_modules_from_strings
    #     import_modules_from_strings(**cfg['custom_imports'])

    ## import modules from plguin/xx, registry will be updated
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
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.environ["WORK_DIR"]#osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.load_from is not None and osp.isfile(args.load_from):
        cfg.load_from = args.load_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    elif args.launcher == 'pytorch':
        torch.backends.cudnn.benchmark = True
        distributed = True
        init_dist(args.launcher, timeout=timedelta(minutes=30), **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    # Create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger_name = 'mmdet'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # seed
    cfg.seed = args.seed
    set_random_seed(args.seed, deterministic=args.deterministic)

    # logger
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=cfg.model.type)
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Set random seed to {args.seed}, 'f'deterministic: {args.deterministic}')


    # Dataset
    datasets = [build_dataset(cfg.data.train)]
    # Save meta info
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmcv_version=mmcv_version, config=cfg.pretty_text, CLASSES=datasets[0].CLASSES, \
                                          PALETTE=datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None) # # for segmentors
    
    # Dataloader
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [build_dataloader(ds,
                        cfg.data.samples_per_gpu,
                        cfg.data.workers_per_gpu,
                        # cfg.gpus will be ignored if distributed
                        len(cfg.gpu_ids),
                        dist=distributed,
                        seed=cfg.seed,
                        # shuffle=False,
                        shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
                        nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
                        runner_type=cfg.runner,
                        ) for ds in datasets
                        ]

    # Model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    print('Total Number of Learnable Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad)/1000/1000, "M")
    if args.fuse_conv_bn:
        model.img_backbone = fuse_conv_bn(model.img_backbone)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    
    model.CLASSES = datasets[0].CLASSES  # add an attribute for visualization convenience
    logger.info(f'Model:\n{model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=find_unused_parameters
                                        )
    else:
        model = DataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # Optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    #optimizer_config = OptimizerHook(**cfg.optimizer_config)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # Runner
    runner = build_runner(cfg.runner, default_args=dict(model=model,
                                                        optimizer=optimizer,
                                                        work_dir=cfg.work_dir,
                                                        logger=logger,
                                                        meta=meta))
    runner.timestamp = timestamp
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    if args.compile_before:
        model.module.img_backbone = torch.compile(model.module.img_backbone, dynamic=False)
        model.module.img_neck = torch.compile(model.module.img_neck, dynamic=False)
        model.module.pts_bbox_head.agent_traj_fus = torch.compile(model.module.pts_bbox_head.agent_traj_fus, dynamic=False)
        model.module.pts_bbox_head.ego_lcf_encoder = torch.compile(model.module.pts_bbox_head.ego_lcf_encoder, dynamic=False)
        model.module.pts_bbox_head.img_position_encoder = torch.compile(model.module.pts_bbox_head.img_position_encoder, dynamic=False)
        model.module.pts_bbox_head.agent_ref_embedding = torch.compile(model.module.pts_bbox_head.agent_ref_embedding, dynamic=False)
        model.module.pts_bbox_head.map_ref_embedding = torch.compile(model.module.pts_bbox_head.map_ref_embedding, dynamic=False)
        model.module.pts_bbox_head.featurized_pe = torch.compile(model.module.pts_bbox_head.featurized_pe, dynamic=False)
        model.module.pts_bbox_head.spatial_alignment = torch.compile(model.module.pts_bbox_head.spatial_alignment, dynamic=False)
        model.module.pts_bbox_head.time_embedding = torch.compile(model.module.pts_bbox_head.time_embedding, dynamic=False)
        model.module.pts_bbox_head.ego_pose_pe = torch.compile(model.module.pts_bbox_head.ego_pose_pe, dynamic=False)
        model.module.pts_bbox_head.ego_pose_memory = torch.compile(model.module.pts_bbox_head.ego_pose_memory, dynamic=False)
    
    
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if args.remove_orig_mod:
            tmp_checkpoint = torch.load(cfg.load_from, map_location="cpu")
            remapped_checkpoint = {"meta":tmp_checkpoint["meta"], "state_dict":{}}
            for k, v in tmp_checkpoint["state_dict"].items():
                new_key = k
                if "_orig_mod" in k:
                    new_key = k.replace("._orig_mod", "")
                remapped_checkpoint["state_dict"][new_key] = v
            torch.save(remapped_checkpoint, str(cfg.load_from[:-4])+"_remapped.pth")
            cfg.load_from = str(cfg.load_from[:-4])+"_remapped.pth"
            print("Remap checkpoint to", str(args.load_from))

        # Added by Claude Code: Use partial loading if requested
        if args.partial_load:
            load_checkpoint_partial(cfg.load_from, model, logger)
        else:
            runner.load_checkpoint(cfg.load_from)
    
    if args.compile_after:
        model.module.img_backbone = torch.compile(model.module.img_backbone, dynamic=False)
        model.module.img_neck = torch.compile(model.module.img_neck, dynamic=False)
        model.module.pts_bbox_head.agent_traj_fus = torch.compile(model.module.pts_bbox_head.agent_traj_fus, dynamic=False)
        model.module.pts_bbox_head.ego_lcf_encoder = torch.compile(model.module.pts_bbox_head.ego_lcf_encoder, dynamic=False)
        model.module.pts_bbox_head.img_position_encoder = torch.compile(model.module.pts_bbox_head.img_position_encoder, dynamic=False)
        model.module.pts_bbox_head.agent_ref_embedding = torch.compile(model.module.pts_bbox_head.agent_ref_embedding, dynamic=False)
        model.module.pts_bbox_head.map_ref_embedding = torch.compile(model.module.pts_bbox_head.map_ref_embedding, dynamic=False)
        model.module.pts_bbox_head.featurized_pe = torch.compile(model.module.pts_bbox_head.featurized_pe, dynamic=False)
        model.module.pts_bbox_head.spatial_alignment = torch.compile(model.module.pts_bbox_head.spatial_alignment, dynamic=False)
        model.module.pts_bbox_head.time_embedding = torch.compile(model.module.pts_bbox_head.time_embedding, dynamic=False)
        model.module.pts_bbox_head.ego_pose_pe = torch.compile(model.module.pts_bbox_head.ego_pose_pe, dynamic=False)
        model.module.pts_bbox_head.ego_pose_memory = torch.compile(model.module.pts_bbox_head.ego_pose_memory, dynamic=False)
    runner.run(data_loaders, cfg.workflow)

if __name__ == '__main__':
    main()
