# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from torch import distributed as dist

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet3d.core.hook import CustomDistEvalHook, CustomEMADistEvalHook, CustomEMAEvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmcv.runner import load_checkpoint
# from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
# from mmseg.core import EvalHook as MMSEG_EvalHook
# from mmseg.datasets import build_dataloader as build_mmseg_dataloader
# from mmseg.utils import get_root_logger as get_mmseg_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resume_with_optimizer_matching(runner, checkpoint_path, logger=None):
    """
    Resume training from a checkpoint with flexible optimizer state loading.
    This handles cases where new parameters are added to the model (e.g., during fine-tuning).
    
    Args:
        runner: The runner object
        checkpoint_path: Path to the checkpoint file
        logger: Logger for printing messages
    """
    if logger is None:
        logger = get_mmdet_root_logger()
    
    logger.info(f"[Fine-tuning Resume] Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Load model weights with non-strict matching (allows new parameters)
    load_state_dict = checkpoint['state_dict']
    model_state_dict = runner.model.state_dict()
    
    # Check for missing and unexpected keys
    missing_keys = []
    unexpected_keys = []
    for key in model_state_dict.keys():
        if key not in load_state_dict:
            missing_keys.append(key)
    for key in load_state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
    
    if missing_keys:
        logger.warning(f"[Fine-tuning Resume] Missing keys in checkpoint (will be randomly initialized):")
        for key in missing_keys:
            logger.warning(f"  - {key}")
    if unexpected_keys:
        logger.warning(f"[Fine-tuning Resume] Unexpected keys in checkpoint (will be ignored):")
        for key in unexpected_keys:
            logger.warning(f"  - {key}")
    
    # Load model state dict with strict=False
    runner.model.load_state_dict(load_state_dict, strict=False)
    logger.info("[Fine-tuning Resume] Model weights loaded successfully")
    
    # 2. Load optimizer state with parameter matching by name
    if 'optimizer' in checkpoint:
        logger.info("[Fine-tuning Resume] Attempting to load optimizer state with parameter matching...")
        
        try:
            checkpoint_optimizer_state = checkpoint['optimizer']
            
            # Build a mapping from parameter name to parameter object for current model
            current_param_name_to_param = {name: param for name, param in runner.model.named_parameters()}
            current_param_to_name = {id(param): name for name, param in runner.model.named_parameters()}
            
            # Get checkpoint state dict
            ckpt_state_dict = checkpoint['state_dict']
            
            # Build mapping from checkpoint parameter names to their optimizer state IDs
            # The optimizer stores state by parameter ID (object id), but we can match by name
            
            # Get current optimizer state dict
            current_optimizer_state_dict = runner.optimizer.state_dict()
            
            # Create a new state dict with matched parameters
            new_optimizer_state = {}
            
            # Map checkpoint parameter IDs to current parameter IDs by matching parameter names
            # First, we need to figure out which checkpoint param ID corresponds to which param name
            
            # Strategy: Reconstruct the parameter ID mapping by iterating through param groups
            # and matching with model parameters by position/name
            
            # Get all parameters in order from current model
            current_model_params = list(runner.model.parameters())
            current_param_ids = [id(p) for p in current_model_params]
            
            # The checkpoint was saved with a previous version of the model
            # We need to match checkpoint optimizer state to current optimizer by parameter names
            
            # Since optimizer state uses parameter IDs which change between runs,
            # we'll use a name-based matching approach:
            
            # Get checkpoint param groups
            ckpt_param_groups = checkpoint_optimizer_state['param_groups']
            ckpt_state = checkpoint_optimizer_state['state']
            
            # Get parameters that were in the checkpoint model (by name)
            checkpoint_param_names = set(ckpt_state_dict.keys())
            current_param_names = set(model_state_dict.keys())
            
            # Find common parameters
            common_param_names = checkpoint_param_names & current_param_names
            
            # For each current parameter, if it existed in checkpoint, copy its optimizer state
            loaded_params = 0
            new_params = 0
            
            # Build mapping from parameter name to checkpoint optimizer state
            # This requires matching the checkpoint param groups with parameter names
            
            # Simpler approach: Match parameters by their position in the model
            # Assumption: The order of parameters hasn't changed for existing parameters
            
            # Get list of checkpoint parameters (those that exist in both models)
            checkpoint_model_params = []
            for name in sorted(ckpt_state_dict.keys()):
                if name in current_param_name_to_param:
                    checkpoint_model_params.append(name)
            
            # Build a mapping from checkpoint param index to current param
            param_index_to_current_id = {}
            param_index = 0
            
            for name in sorted(ckpt_state_dict.keys()):
                if name in current_param_name_to_param:
                    current_param = current_param_name_to_param[name]
                    current_param_id = id(current_param)
                    param_index_to_current_id[param_index] = current_param_id
                param_index += 1
            
            # Now map checkpoint optimizer state to current optimizer state
            # The checkpoint optimizer state keys are old parameter IDs
            # We need to map them to current parameter IDs
            
            # Iterate through checkpoint param groups and match by index
            ckpt_param_index = 0
            for group_idx, ckpt_group in enumerate(ckpt_param_groups):
                for ckpt_param_id in ckpt_group['params']:
                    # This checkpoint param ID corresponds to parameter at index ckpt_param_index
                    if ckpt_param_index in param_index_to_current_id:
                        current_param_id = param_index_to_current_id[ckpt_param_index]
                        
                        # Copy optimizer state from checkpoint to current
                        if ckpt_param_id in ckpt_state:
                            new_optimizer_state[current_param_id] = ckpt_state[ckpt_param_id]
                            loaded_params += 1
                    
                    ckpt_param_index += 1
            
            # Count new parameters
            for param_id in current_param_ids:
                if param_id not in new_optimizer_state:
                    new_params += 1
            
            # Update optimizer state dict
            current_optimizer_state_dict['state'] = new_optimizer_state
            
            # Load the updated state dict
            try:
                runner.optimizer.load_state_dict(current_optimizer_state_dict)
                logger.info(f"[Fine-tuning Resume] Optimizer state loaded: {loaded_params} params from checkpoint, "
                           f"{new_params} new params initialized with default optimizer state")
            except Exception as e:
                logger.warning(f"[Fine-tuning Resume] Failed to load optimizer state dict: {e}")
                logger.warning("[Fine-tuning Resume] Using fresh optimizer for all parameters")
            
        except Exception as e:
            logger.warning(f"[Fine-tuning Resume] Failed to load optimizer state with parameter matching: {e}")
            logger.warning("[Fine-tuning Resume] Skipping optimizer state loading. Using fresh optimizer.")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("[Fine-tuning Resume] No optimizer state found in checkpoint")
    
    # 3. Load meta information (epoch, iter, etc.)
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        if 'epoch' in meta:
            runner._epoch = meta['epoch']
            logger.info(f"[Fine-tuning Resume] Resuming from epoch {meta['epoch'] + 1}")
        if 'iter' in meta:
            runner._iter = meta['iter']
        if 'inner_iter' in meta:
            runner._inner_iter = meta.get('inner_iter', 0)
    
    # 4. Load learning rate scheduler state if available
    if hasattr(runner, 'lr_scheduler') and runner.lr_scheduler is not None:
        if 'lr_scheduler' in checkpoint:
            try:
                runner.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                logger.info("[Fine-tuning Resume] LR scheduler state loaded")
            except Exception as e:
                logger.warning(f"[Fine-tuning Resume] Failed to load LR scheduler state: {e}")
    
    logger.info("[Fine-tuning Resume] Resume complete!")


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = CustomDistEvalHook if distributed else MMDET_EvalHook
        # eval_hook = CustomEMADistEvalHook if distributed else CustomEMAEvalHook
        # eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # Priority 1: If auto_resume is enabled, check for latest checkpoint first
    # Priority 2: Fall back to resume_from if no latest checkpoint found
    resume_from = None
    
    if cfg.get('auto_resume'):
        # Try to find latest checkpoint in work_dir
        latest_checkpoint = find_latest_checkpoint(cfg.work_dir)
        if latest_checkpoint is not None:
            print(f"[Auto-Resume] Found latest checkpoint: {latest_checkpoint}")
            resume_from = latest_checkpoint
        elif cfg.resume_from is not None:
            print(f"[Auto-Resume] No latest checkpoint found, using base checkpoint: {cfg.resume_from}")
            resume_from = cfg.resume_from
        else:
            print("[Auto-Resume] No checkpoint found, starting from scratch")
    elif cfg.resume_from is not None:
        # Auto-resume disabled, use resume_from directly
        print(f"[Resume] Using specified checkpoint: {cfg.resume_from}")
        resume_from = cfg.resume_from

    if resume_from is not None:
        cfg.resume_from = resume_from
    
    # Handle checkpoint loading with flexible optimizer matching for fine-tuning
    if cfg.resume_from:
        # Check if we should use flexible resume (for fine-tuning with new parameters)
        use_flexible_resume = cfg.get('flexible_resume', False)
        
        if use_flexible_resume:
            # Use our custom resume function that handles optimizer state mismatch
            resume_with_optimizer_matching(runner, cfg.resume_from, logger)
        else:
            # Try normal resume first
            try:
                runner.resume(cfg.resume_from)
                logger.info("[Resume] Checkpoint loaded successfully with standard resume")
            except ValueError as e:
                if "different number of parameter groups" in str(e):
                    # Optimizer parameter mismatch detected - switch to flexible resume
                    logger.warning(f"[Resume] Standard resume failed: {e}")
                    logger.warning("[Resume] Detected parameter mismatch - switching to flexible resume mode")
                    logger.warning("[Resume] This is expected when fine-tuning with new model components")
                    resume_with_optimizer_matching(runner, cfg.resume_from, logger)
                else:
                    # Re-raise other errors
                    raise
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    runner.run(data_loaders, cfg.workflow)

def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)
