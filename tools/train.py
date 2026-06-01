# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
# from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
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
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='force CPU-only mode for debugging (no CUDA required)')
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

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Handle CPU-only debug mode
    if args.cpu_only:
        # Force CUDA availability to False
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Set GPU IDs to empty to force CPU mode
        cfg.gpu_ids = []
        # Disable CUDNN benchmark
        cfg.cudnn_benchmark = False
        print("DEBUG MODE: Forced CPU-only execution")
            
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        # Set environment variable so hooks can access it
        os.environ['RESUME_FROM_CHECKPOINT'] = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        if not args.cpu_only:
            cfg.gpu_ids = range(1)
            warnings.warn('`--gpus` is deprecated because we only support '
                          'single GPU mode in non-distributed training. '
                          'Use `gpus=1` now.')
        else:
            cfg.gpu_ids = []
    if args.gpu_ids is not None:
        if not args.cpu_only:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed training. Use the first GPU '
                          'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = []
    if args.gpus is None and args.gpu_ids is None and not args.cpu_only:
        cfg.gpu_ids = [args.gpu_id]
    elif args.gpus is None and args.gpu_ids is None and args.cpu_only:
        cfg.gpu_ids = []

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    
    # Store resume_from in meta so hooks can access it
    if hasattr(cfg, 'resume_from') and cfg.resume_from:
        meta['resume_from'] = cfg.resume_from

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            # mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # EDIT:
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
        torch.autograd.detect_anomaly()

    # Train the model  
    if args.cpu_only:
        # Simple CPU-only training implementation
        from mmcv.runner import build_optimizer, build_runner
        from mmcv.parallel import MMDataParallel
        from mmdet.datasets.builder import build_dataloader as build_mmdet_dataloader
        
        # Fix dataloader batch size calculation
        original_gpu_ids = cfg.gpu_ids
        cfg.gpu_ids = [0]
        
        dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
        runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
        
        data_loaders = [
            build_mmdet_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=distributed,
                seed=cfg.seed,
                runner_type=runner_type,
                persistent_workers=cfg.data.get('persistent_workers', False))
            for ds in dataset
        ]
        
        # Restore empty gpu_ids and keep model on CPU
        cfg.gpu_ids = original_gpu_ids
        
        # Handle CPU-only case - don't wrap with MMDataParallel if no GPUs
        if cfg.gpu_ids:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            # For CPU-only, we can use the model directly or wrap with MMDataParallel with no device_ids
            model = MMDataParallel(model, device_ids=None)
        
        # Standard training setup  
        optimizer = build_optimizer(model, cfg.optimizer)
        
        # Check if debug mode with limited iterations
        debug_max_iters = cfg.get('debug_max_iters', None)
        if debug_max_iters:
            # For quick debug: limit the runner to specified iterations
            runner_cfg = dict(
                type='IterBasedRunner',
                max_iters=debug_max_iters,
            )
            print(f"DEBUG MODE: Limited to {debug_max_iters} iterations for quick testing")
        else:
            runner_cfg = cfg.runner
            
        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
        
        runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                     cfg.checkpoint_config, cfg.log_config,
                                     cfg.get('momentum_config', None))
        
        # # Handle checkpoint loading with optional non-strict mode (for fine-tuning)
        # # When checkpoint_strict_load=False, missing keys (e.g., DINO components) are allowed
        # strict_load = cfg.get('checkpoint_strict_load', True)  # Default to strict loading
        
        # if cfg.resume_from:
        #     if not strict_load:
        #         # For fine-tuning: load ONLY model weights, skip optimizer/scheduler state
        #         # This allows adding new components (like DINO) that weren't in base checkpoint
        #         # and avoids optimizer state mismatch errors
        #         import torch
        #         from mmcv.runner import load_checkpoint
                
        #         print(f"[Fine-tuning Mode] Loading checkpoint for fine-tuning: {cfg.resume_from}")
                
        #         # Load checkpoint to extract metadata
        #         checkpoint = torch.load(cfg.resume_from, map_location='cpu')
                
        #         # Load model weights with strict=False (allows missing DINO keys)
        #         load_checkpoint(model, cfg.resume_from, map_location='cpu', strict=False)
        #         print("[Fine-tuning Mode] Model weights loaded. Missing keys (DINO) randomly initialized.")
                
        #         # Resume epoch number and iteration count (but NOT optimizer state)
        #         if 'meta' in checkpoint and 'epoch' in checkpoint['meta']:
        #             start_epoch = checkpoint['meta']['epoch']
        #             runner._epoch = start_epoch
        #             runner._iter = checkpoint['meta'].get('iter', 0)
        #             runner._inner_iter = checkpoint['meta'].get('inner_iter', 0)
        #             print(f"[Fine-tuning Mode] Resuming from epoch {start_epoch + 1}")
        #         else:
        #             print("[Fine-tuning Mode] No epoch info in checkpoint - starting from epoch 1")
                
        #         print("[Fine-tuning Mode] Optimizer state NOT loaded - using fresh optimizer with config LR.")
        #         # Note: Optimizer will use the LR from config (not from checkpoint)
        #         # This allows you to use a different LR schedule for fine-tuning
        #     else:
        #         # Normal resume with strict matching (loads model + optimizer + scheduler)
        #         runner.resume(cfg.resume_from)
        # elif cfg.load_from:
        #     runner.load_checkpoint(cfg.load_from, strict=strict_load)
        
        # original:
        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)

            
        runner.run(data_loaders, cfg.workflow)
        
    else:
        train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)


if __name__ == '__main__':
    main()
