import os.path as osp
import torch
import numpy as np
from typing import Optional
from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner.hooks.hook import HOOKS


@HOOKS.register_module()
class WeightMonitorHook(LoggerHook):
    """Hook to monitor model weights, biases, and their statistics.
    
    This hook tracks:
    - Weight and bias norms (L1, L2)
    - Weight and bias statistics (mean, std, min, max)
    - Gradient norms
    - Weight update ratios
    
    Args:
        log_dir (str, optional): Directory to save tensorboard logs.
        interval (int): Logging interval. Default: 10.
        monitor_layers (list, optional): Specific layer names to monitor. 
            If None, monitors all layers with weights.
        track_weight_updates (bool): Whether to track weight update ratios.
        min_norm_threshold (float): Minimum norm threshold for logging.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 monitor_layers: Optional[list] = None,
                 track_weight_updates: bool = True,
                 min_norm_threshold: float = 1e-8,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir
        self.monitor_layers = monitor_layers
        self.track_weight_updates = track_weight_updates
        self.min_norm_threshold = min_norm_threshold
        self.prev_weights = {}

    def before_run(self, runner) -> None:
        super().before_run(runner)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install tensorboard" to install '
                'the dependencies to use torch.utils.tensorboard')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'weight_monitor_logs')
        self.writer = SummaryWriter(self.log_dir)
        
        # Store initial weights for update ratio tracking
        if self.track_weight_updates:
            self._store_weights(runner.model)

    def _store_weights(self, model):
        """Store current weights for update ratio calculation."""
        self.prev_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                self.prev_weights[name] = param.data.clone()

    def _should_monitor_layer(self, name):
        """Check if a layer should be monitored."""
        if self.monitor_layers is None:
            return True
        return any(layer in name for layer in self.monitor_layers)

    def _compute_stats(self, tensor):
        """Compute statistics for a tensor."""
        if tensor.numel() == 0:
            return {}
        
        tensor_flat = tensor.view(-1)
        return {
            'mean': tensor_flat.mean().item(),
            'std': tensor_flat.std().item(),
            'min': tensor_flat.min().item(),
            'max': tensor_flat.max().item(),
            'l1_norm': tensor_flat.abs().sum().item(),
            'l2_norm': tensor_flat.norm().item(),
            'zero_fraction': (tensor_flat == 0).float().mean().item()
        }

    def log(self, runner) -> None:
        # Log standard training metrics
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

        # Monitor model weights and biases
        current_iter = self.get_iter(runner)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        for name, param in model.named_parameters():
            if not self._should_monitor_layer(name) or not param.requires_grad:
                continue
                
            if param.data is not None:
                # Compute weight statistics
                stats = self._compute_stats(param.data)
                
                # Log weight statistics
                for stat_name, stat_val in stats.items():
                    if stat_val > self.min_norm_threshold or stat_name in ['mean', 'std']:
                        self.writer.add_scalar(f'weights/{name}/{stat_name}', 
                                             stat_val, current_iter)
                
                # Track weight updates if enabled
                if self.track_weight_updates and name in self.prev_weights:
                    weight_diff = param.data - self.prev_weights[name]
                    update_stats = self._compute_stats(weight_diff)
                    
                    # Log update statistics
                    for stat_name, stat_val in update_stats.items():
                        if stat_val > self.min_norm_threshold:
                            self.writer.add_scalar(f'weight_updates/{name}/{stat_name}', 
                                                 stat_val, current_iter)
                    
                    # Log update ratio (||update|| / ||weight||)
                    weight_norm = stats['l2_norm']
                    update_norm = update_stats['l2_norm']
                    if weight_norm > self.min_norm_threshold:
                        update_ratio = update_norm / weight_norm
                        self.writer.add_scalar(f'weight_update_ratios/{name}', 
                                             update_ratio, current_iter)

            # Log gradients if available
            if param.grad is not None:
                grad_stats = self._compute_stats(param.grad.data)
                for stat_name, stat_val in grad_stats.items():
                    if stat_val > self.min_norm_threshold:
                        self.writer.add_scalar(f'gradients/{name}/{stat_name}', 
                                             stat_val, current_iter)

        # Update stored weights for next iteration
        if self.track_weight_updates:
            self._store_weights(model)

        # Log layer-wise statistics
        self._log_layer_summary(model, current_iter)

    def _log_layer_summary(self, model, current_iter):
        """Log summary statistics by layer type."""
        layer_stats = {}
        
        for name, module in model.named_modules():
            if not self._should_monitor_layer(name):
                continue
                
            module_type = type(module).__name__
            if module_type not in layer_stats:
                layer_stats[module_type] = {
                    'weight_norms': [],
                    'bias_norms': [],
                    'grad_norms': []
                }
            
            # Collect weight norms
            if hasattr(module, 'weight') and module.weight is not None:
                weight_norm = module.weight.data.norm().item()
                layer_stats[module_type]['weight_norms'].append(weight_norm)
                
                if module.weight.grad is not None:
                    grad_norm = module.weight.grad.data.norm().item()
                    layer_stats[module_type]['grad_norms'].append(grad_norm)
            
            # Collect bias norms
            if hasattr(module, 'bias') and module.bias is not None:
                bias_norm = module.bias.data.norm().item()
                layer_stats[module_type]['bias_norms'].append(bias_norm)

        # Log aggregated statistics
        for layer_type, stats in layer_stats.items():
            for stat_type, values in stats.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.writer.add_scalar(f'layer_summary/{layer_type}/{stat_type}_mean', 
                                         mean_val, current_iter)
                    self.writer.add_scalar(f'layer_summary/{layer_type}/{stat_type}_std', 
                                         std_val, current_iter)

    def after_run(self, runner) -> None:
        self.writer.close()
