import bisect
import os.path as osp
import torch

from torch.utils.data import DataLoader
import torch.distributed as dist
from mmcv.runner import DistEvalHook, EvalHook, load_checkpoint
from mmdet.core.evaluation.eval_hooks import _calc_dynamic_intervals
from torch.nn.modules.batchnorm import _BatchNorm

class CustomEMADistEvalHook(DistEvalHook):
    """Distributed evaluation hook that evaluates EMA checkpoint instead of regular model."""

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomEMADistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _get_ema_checkpoint_path(self, runner):
        """Get the path to the current epoch's EMA checkpoint."""
        epoch_num = runner.epoch + 1
        ema_checkpoint_name = f'epoch_{epoch_num}_ema.pth'
        ema_checkpoint_path = osp.join(runner.work_dir, ema_checkpoint_name)
        return ema_checkpoint_path

    def _load_ema_weights_to_model(self, runner):
        """Load EMA weights into the model for evaluation."""
        ema_checkpoint_path = self._get_ema_checkpoint_path(runner)
        
        if not osp.exists(ema_checkpoint_path):
            runner.logger.warning(f'EMA checkpoint not found at {ema_checkpoint_path}, using regular model weights for evaluation')
            return False
            
        try:
            # Load EMA checkpoint
            ema_checkpoint = torch.load(ema_checkpoint_path, map_location='cpu')
            ema_state_dict = ema_checkpoint['state_dict']
            
            # Store current model state for restoration later
            self._original_state_dict = runner.model.state_dict()
            
            # Handle state_dict key mismatch between EMA and model wrapper
            # Check if the current model is wrapped (has 'module.' prefix)
            current_keys = list(self._original_state_dict.keys())
            ema_keys = list(ema_state_dict.keys())
            
            # Determine if we need to add or remove 'module.' prefix
            current_has_module = any(key.startswith('module.') for key in current_keys)
            ema_has_module = any(key.startswith('module.') for key in ema_keys)
            
            if current_has_module and not ema_has_module:
                # Current model has module prefix, EMA doesn't - add module prefix to EMA
                adjusted_ema_state_dict = {}
                for key, value in ema_state_dict.items():
                    adjusted_ema_state_dict[f'module.{key}'] = value
                ema_state_dict = adjusted_ema_state_dict
                runner.logger.info('Added "module." prefix to EMA state dict for compatibility')
            elif not current_has_module and ema_has_module:
                # Current model doesn't have module prefix, EMA does - remove module prefix from EMA
                adjusted_ema_state_dict = {}
                for key, value in ema_state_dict.items():
                    if key.startswith('module.'):
                        adjusted_ema_state_dict[key[7:]] = value  # Remove 'module.' prefix
                    else:
                        adjusted_ema_state_dict[key] = value
                ema_state_dict = adjusted_ema_state_dict
                runner.logger.info('Removed "module." prefix from EMA state dict for compatibility')
            
            # Load EMA weights into model
            runner.model.load_state_dict(ema_state_dict, strict=True)
            runner.logger.info(f'Loaded EMA weights from {ema_checkpoint_path} for evaluation')
            return True
            
        except Exception as e:
            runner.logger.error(f'Failed to load EMA checkpoint {ema_checkpoint_path}: {e}')
            return False

    def _restore_original_weights(self, runner):
        """Restore original model weights after evaluation."""
        if hasattr(self, '_original_state_dict'):
            runner.model.load_state_dict(self._original_state_dict)
            delattr(self, '_original_state_dict')
            runner.logger.info('Restored original model weights after EMA evaluation')

    def _do_evaluate(self, runner):
        """perform evaluation using EMA checkpoint and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        if runner.rank == 0:
            # Load EMA weights into model
            ema_loaded = self._load_ema_weights_to_model(runner)
            
            try:
                from mmdet3d.apis import custom_single_gpu_test
                results = custom_single_gpu_test(runner.model, self.dataloader)
                self.latest_results = results
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                key_score = self.evaluate(runner, results)

                # Log that this evaluation used EMA weights
                if ema_loaded:
                    runner.logger.info(f'Evaluation completed using EMA checkpoint from epoch {runner.epoch + 1}')
                else:
                    runner.logger.info(f'Evaluation completed using regular model weights (EMA checkpoint not available)')
                        
                # the key_score may be `None` so it needs to skip the action to save
                # the best checkpoint
                if self.save_best and key_score:
                    self._save_ckpt(runner, key_score)
                    
            finally:
                # Always restore original weights regardless of success or failure
                if ema_loaded:
                    self._restore_original_weights(runner)
                
        dist.barrier()


class CustomEMAEvalHook(EvalHook):
    """Non-distributed evaluation hook that evaluates EMA checkpoint instead of regular model."""

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomEMAEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _get_ema_checkpoint_path(self, runner):
        """Get the path to the current epoch's EMA checkpoint."""
        epoch_num = runner.epoch + 1
        ema_checkpoint_name = f'epoch_{epoch_num}_ema.pth'
        ema_checkpoint_path = osp.join(runner.work_dir, ema_checkpoint_name)
        return ema_checkpoint_path

    def _load_ema_weights_to_model(self, runner):
        """Load EMA weights into the model for evaluation."""
        ema_checkpoint_path = self._get_ema_checkpoint_path(runner)
        
        if not osp.exists(ema_checkpoint_path):
            runner.logger.warning(f'EMA checkpoint not found at {ema_checkpoint_path}, using regular model weights for evaluation')
            return False
            
        try:
            # Load EMA checkpoint
            ema_checkpoint = torch.load(ema_checkpoint_path, map_location='cpu')
            ema_state_dict = ema_checkpoint['state_dict']
            
            # Store current model state for restoration later
            self._original_state_dict = runner.model.state_dict()
            
            # Handle state_dict key mismatch between EMA and model wrapper
            # Check if the current model is wrapped (has 'module.' prefix)
            current_keys = list(self._original_state_dict.keys())
            ema_keys = list(ema_state_dict.keys())
            
            # Determine if we need to add or remove 'module.' prefix
            current_has_module = any(key.startswith('module.') for key in current_keys)
            ema_has_module = any(key.startswith('module.') for key in ema_keys)
            
            if current_has_module and not ema_has_module:
                # Current model has module prefix, EMA doesn't - add module prefix to EMA
                adjusted_ema_state_dict = {}
                for key, value in ema_state_dict.items():
                    adjusted_ema_state_dict[f'module.{key}'] = value
                ema_state_dict = adjusted_ema_state_dict
                runner.logger.info('Added "module." prefix to EMA state dict for compatibility')
            elif not current_has_module and ema_has_module:
                # Current model doesn't have module prefix, EMA does - remove module prefix from EMA
                adjusted_ema_state_dict = {}
                for key, value in ema_state_dict.items():
                    if key.startswith('module.'):
                        adjusted_ema_state_dict[key[7:]] = value  # Remove 'module.' prefix
                    else:
                        adjusted_ema_state_dict[key] = value
                ema_state_dict = adjusted_ema_state_dict
                runner.logger.info('Removed "module." prefix from EMA state dict for compatibility')
            
            # Load EMA weights into model
            runner.model.load_state_dict(ema_state_dict, strict=True)
            runner.logger.info(f'Loaded EMA weights from {ema_checkpoint_path} for evaluation')
            return True
            
        except Exception as e:
            runner.logger.error(f'Failed to load EMA checkpoint {ema_checkpoint_path}: {e}')
            return False

    def _restore_original_weights(self, runner):
        """Restore original model weights after evaluation."""
        if hasattr(self, '_original_state_dict'):
            runner.model.load_state_dict(self._original_state_dict)
            delattr(self, '_original_state_dict')
            runner.logger.info('Restored original model weights after EMA evaluation')

    def _do_evaluate(self, runner):
        """perform evaluation using EMA checkpoint and save ckpt."""
        if not self._should_evaluate(runner):
            return

        # Load EMA weights into model
        ema_loaded = self._load_ema_weights_to_model(runner)
        
        try:
            from mmdet3d.apis import custom_single_gpu_test
            results = custom_single_gpu_test(runner.model, self.dataloader)
            self.latest_results = results
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # Log that this evaluation used EMA weights
            if ema_loaded:
                runner.logger.info(f'Evaluation completed using EMA checkpoint from epoch {runner.epoch + 1}')
            else:
                runner.logger.info(f'Evaluation completed using regular model weights (EMA checkpoint not available)')
                    
            # the key_score may be `None` so it needs to skip the action to save
            # the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
                
        finally:
            # Always restore original weights regardless of success or failure
            if ema_loaded:
                self._restore_original_weights(runner)