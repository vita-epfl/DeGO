import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_loss, build_head
from mmdet3d.models.dego_modules.utils import move_gaussians_temporal_module
from mmdet3d.models.dego_modules.feature_distillation import FeatureProjector, FeatureDistillationLoss, TemporalFeatureContrastiveLoss
from .mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn.bricks.transformer import build_feedforward_network
from gsplat import quat_scale_to_covar_preci

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Imported from pytorch3d https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_raw_multiply"""
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

@DETECTORS.register_module()
class DeGOVGGT(MVXTwoStageDetector):

    def __init__(self,
                 num_classes=16,
                 with_others=False,
                 in_channels=128,
                 rasterizer=None,
                 temporal_module=None,
                 gaussian_decoder=None,
                 voxel_grid_cfg=None,
                 eval_threshold_range=[.1],
                 gaussian_init_scale=2,
                 num_gaussians=10000,
                 max_neighborhood=4,
                 use_opacity=True,
                 use_scale=True,
                 use_rotation=True,
                 scale_act=True,
                 render_semantic=True,
                 render_depth=True,
                 render_rgb=False,
                 sh_degree=0,
                 initial_mean=True,
                 scale_range=None, # [0.05, .32],
                 temporal_frame_ids=[0],
                 scale_multiplier=1.,
                 move_dynamic_gaussians=False,
                 use_movement_reg=False,
                 num_head_layers=1,
                 use_mask=True,
                 loss_occ_density=None,
                 loss_occ_semantics=None,
                 temporal_loss_3d=False,
                 train_cfg=None,
                 # VGGT distillation parameters
                 use_vggt_distillation=False,
                 vggt_distillation_cfg=None,
                 # VGGT feature rendering distillation (Feature3DGS-style)
                 use_vggt_feature_distillation=False,
                 vggt_feature_distillation_config=None,
                 vggt_feature_distillation_start_epoch=1,
                 **kwargs):
        super(DeGOVGGT, self).__init__(**kwargs)
        self.pts_bbox_head = None
        self.with_others = with_others
        self.num_classes = num_classes + 1 if with_others else num_classes
        self.eval_threshold_range = eval_threshold_range
        self.voxel_grid_cfg = voxel_grid_cfg
        self.max_neighborhood = max_neighborhood # in each direction in 3D
        self.voxel_centers = None
        self.prev_feat = None
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])
        self.temporal_frame_ids = torch.tensor(temporal_frame_ids)
        self.zero_index = [i for i, t in enumerate(temporal_frame_ids) if t==0][0]
        self.train_cfg = train_cfg

        self.gaussian_decoder = build_feedforward_network(gaussian_decoder) if gaussian_decoder is not None else None
        self.rasterizer = build_head(rasterizer) if rasterizer is not None else None
        self.move_dynamic_gaussians = move_dynamic_gaussians
        self.next_t_index = [i for i, t in enumerate(temporal_frame_ids) if t==1][0] if 1 in temporal_frame_ids else None

        self.scale_multiplier = scale_multiplier

        # Initial gaussian properties
        if num_gaussians is None:
            sparse_grid_cfg = self.voxel_grid_cfg.deepcopy()
            sparse_grid_cfg['x'][2] *= gaussian_init_scale
            sparse_grid_cfg['y'][2] *= gaussian_init_scale
            sparse_grid_cfg['z'][2] *= gaussian_init_scale
            self.initial_means = nn.Parameter(self.create_voxel_centers(sparse_grid_cfg).flatten(0, -2), requires_grad=initial_mean)
        else:
            self.initial_means = nn.Parameter(self.create_voxel_centers_random(self.voxel_grid_cfg, num_gaussians), requires_grad=initial_mean)
        # Depth
        self.render_depth = render_depth

        # Semantic Head
        self.render_semantic = render_semantic
        semantic_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            semantic_layers.append(nn.Linear(in_channels*2, in_channels*2))
            semantic_layers.append(nn.LeakyReLU(inplace=True))
        semantic_layers.append(nn.Linear(in_channels*2, self.num_classes))
        self.semantic_head = nn.Sequential(*semantic_layers)

        # RGB Head
        self.render_rgb = render_rgb
        self.sh_degree = sh_degree
        rgb_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            rgb_layers.append(nn.Linear(in_channels*2, in_channels*2))
            rgb_layers.append(nn.LeakyReLU(inplace=True))
        rgb_layers.append(nn.Linear(in_channels*2, 3*((sh_degree+1)**2)))
        self.rgb_head = nn.Sequential(*rgb_layers) if self.render_rgb else None

        # Opacity Head
        opacity_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            opacity_layers.append(nn.Linear(in_channels*2, in_channels*2))
            opacity_layers.append(nn.LeakyReLU(inplace=True))
        opacity_layers.append(nn.Linear(in_channels*2, 1))
        opacity_layers.append(nn.Sigmoid())
        self.opacity_head = nn.Sequential(*opacity_layers) if use_opacity else None

        # Scale Head
        scale_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            scale_layers.append(nn.Linear(in_channels*2, in_channels*2))
            scale_layers.append(nn.LeakyReLU(inplace=True))
        scale_layers.append(nn.Linear(in_channels*2, 3))
        if scale_act:
            scale_layers.append(nn.Sigmoid())
        self.scale_head = nn.Sequential(*scale_layers) if use_scale else None
        self.scale_range = scale_range

        # Rotation Head
        rotation_layers = [nn.Linear(in_channels, in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers):
            rotation_layers.append(nn.Linear(in_channels*2, in_channels*2))
            # rotation_layers.append(nn.LayerNorm(in_channels))
            rotation_layers.append(nn.LeakyReLU(inplace=True))
        rotation_layers.append(nn.Linear(in_channels*2, 4))
        self.rotation_head = nn.Sequential(*rotation_layers) if use_rotation else None

        # Learnable query vector
        self.gaussian_queries = nn.Parameter(torch.empty(self.initial_means.shape[0], in_channels))
        nn.init.normal_(self.gaussian_queries)

        # Temporal Module
        self.temporal_module = build_feedforward_network(temporal_module) if temporal_module is not None else None
        self.movement_regularizer = build_loss(dict(type='MovementRegularizer')) if temporal_module is not None and use_movement_reg else None

        # 3D loss
        self.use_mask = use_mask
        self.temporal_loss_3d = temporal_loss_3d
        self.loss_occ_density = build_loss(loss_occ_density) if loss_occ_density is not None else None
        self.loss_occ_semantics = build_loss(loss_occ_semantics) if loss_occ_semantics is not None else None

        # VGGT distillation setup - LAZY INITIALIZATION
        # Only initialize VGGT teacher during training, not during inference/validation
        self.use_vggt_distillation = use_vggt_distillation
        self.vggt_distillation_cfg = vggt_distillation_cfg
        self.vggt_initialized = False  # Track if VGGT has been initialized
        self.vggt_model = None  # Will be loaded lazily only if needed
        
        # VGGT feature rendering distillation setup
        if vggt_feature_distillation_config is None:
            vggt_feature_distillation_config = {}
        vggt_feature_distillation_start_epoch = vggt_feature_distillation_config.get(
            'start_epoch', vggt_feature_distillation_start_epoch)
        self.use_vggt_feature_distillation = use_vggt_feature_distillation
        self.vggt_feature_distillation_config = vggt_feature_distillation_config
        self.vggt_feature_initialized = False
        self.vggt_feature_cache_ready = False
        self.vggt_feature_distillation_start_epoch = int(vggt_feature_distillation_start_epoch)
        self.current_epoch = 1
        self.current_iter = 0
        self._vggt_distillation_wait_logged = False
        self.feature_head = None  # Per-Gaussian feature head
        self.feature_projector = None  # Project VGGT features to compact space
        self.feature_distillation_loss = None  # Feature alignment loss
        self.temporal_feature_loss = None  # Optional temporal contrastive loss
        
        # Store in_channels for lazy initialization
        self.in_channels = in_channels
        
        # Initialize cache-related attributes early if distillation is enabled
        if use_vggt_distillation and vggt_distillation_cfg:
            self.patch_size = 14
            self.use_vggt_cache = vggt_distillation_cfg.get('use_cache', True)
            self.vggt_layer_indices = vggt_distillation_cfg.get('layer_indices', [6, 12, 18])
            self.cache_selected_only = vggt_distillation_cfg.get('cache_selected_only', False)
            self.cache_use_fp16 = vggt_distillation_cfg.get('cache_use_fp16', False)
            self.cache_compress = vggt_distillation_cfg.get('cache_compress', False)
        
        # Initialize cache for VGGT feature distillation if enabled
        if use_vggt_feature_distillation:
            self.patch_size = 14
            self.use_vggt_cache = vggt_feature_distillation_config.get('use_cache', True)
            self.vggt_cache_max_size = vggt_feature_distillation_config.get('cache_max_size', 1000)
            self.vggt_cache_dir = vggt_feature_distillation_config.get('cache_dir', 'data/vggt_cache')
            self.vggt_layer_indices = vggt_feature_distillation_config.get('layer_indices', [20])
            self.cache_selected_only = vggt_feature_distillation_config.get('cache_selected_only', True)
            self.cache_use_fp16 = vggt_feature_distillation_config.get('cache_use_fp16', False)
            self.cache_compress = vggt_feature_distillation_config.get('cache_compress', False)
            self.vggt_feature_cache = OrderedDict()
            self.vggt_teacher_channels = 2048  # VGGT aggregator output
            # Build trainable distillation modules before the optimizer is created.
            # Cache probing and teacher feature loading remain lazy until the start epoch.
            self._init_vggt_feature_distillation(
                vggt_feature_distillation_config,
                load_cache=False)
            self.vggt_feature_initialized = True

    def set_epoch(self, epoch, iteration=None):
        """Receive 1-based epoch information from the training hook."""
        self.current_epoch = int(epoch)
        if iteration is not None:
            self.current_iter = int(iteration)

    def _is_vggt_feature_distillation_active(self):
        if not (self.training and self.use_vggt_feature_distillation):
            return False
        return self.current_epoch >= self.vggt_feature_distillation_start_epoch

    def _get_vggt_feature_distillation_warmup_loss(self):
        """Keep delayed distillation parameters visible to DDP with zero loss."""
        warmup_loss = None
        for module in (self.feature_head, self.feature_projector):
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                param_loss = param.sum() * 0.0
                if warmup_loss is None:
                    warmup_loss = param_loss
                else:
                    warmup_loss = warmup_loss + param_loss
        return warmup_loss

    def _ensure_vggt_initialized(self):
        """Lazy initialization of VGGT - only initialize when actually needed during training.
        
        This prevents loading the heavy VGGT teacher model during inference/validation.
        Also skips loading if cache is complete (all features already cached).
        """
        if not self.vggt_initialized and self.use_vggt_distillation:
            # Always initialize the distillation infrastructure (cache, loss, etc.)
            self._init_vggt_distillation(self.vggt_distillation_cfg)
            
            # But keep vggt_model = None if cache is enabled
            # It will be loaded on-demand only on cache miss
            if self.use_vggt_cache:
                print("[VGGT Distillation] Cache enabled - VGGT model will only load if cache misses occur")
                self.vggt_model = None  # Will be loaded lazily on first cache miss
            else:
                # No cache - must load VGGT model now
                print("[VGGT Distillation] No cache - loading VGGT model...")
                self._load_vggt_model()
            
            self.vggt_initialized = True

    def _load_vggt_model(self):
        """Load the VGGT model itself (separated from distillation setup).
        
        This allows us to load the model on-demand only when cache misses occur.
        """
        from vggt.models.vggt import VGGT
        from .vggt_utils import freeze_model
        
        print("[VGGT Distillation] Loading VGGT teacher model...")
        self.vggt_model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        try:
            state_dict = torch.hub.load_state_dict_from_url(_URL, progress=True)
            self.vggt_model.load_state_dict(state_dict)
            print("[VGGT Distillation] Successfully loaded VGGT model weights")
        except Exception as e:
            print(f"[VGGT Distillation] Warning: Could not load VGGT weights: {e}")
        
        freeze_model(self.vggt_model)
        print("[VGGT Distillation] VGGT teacher model frozen")
        
        
        ## Need to double check when cache missing.
        # # Move VGGT model to the same device as the main model
        # device = next(self.parameters()).device
        # self.vggt_model = self.vggt_model.to(device)
        
        # freeze_model(self.vggt_model)
        # print(f"[VGGT Distillation] VGGT teacher model frozen and moved to {device}")

    def _init_vggt_distillation(self, vggt_cfg):
        """Initialize VGGT distillation components (loss, cache, etc).
        
        Note: VGGT model itself is loaded lazily on first cache miss to save memory.
        
        Args:
            vggt_cfg (dict): Configuration for VGGT distillation
        """
        if vggt_cfg is None:
            vggt_cfg = {}
        
        # VGGT settings
        self.patch_size = 14
        self.vggt_layer_indices = vggt_cfg.get('layer_indices', [6, 12, 18])  # Select 3 layers
        self.vggt_teacher_channels = 2048  # VGGT aggregator output channels
        
        # Feature caching for VGGT (huge speedup during training)
        self.use_vggt_cache = vggt_cfg.get('use_cache', True)  # Enable by default
        self.vggt_cache_max_size = vggt_cfg.get('cache_max_size', 1000)  # Limit cache size
        self.vggt_cache_dir = vggt_cfg.get('cache_dir', 'data/vggt_cache')  # Disk cache directory
        
        # Cache optimization options (to save disk space)
        self.cache_selected_only = vggt_cfg.get('cache_selected_only', False)  # Cache only selected layers
        self.cache_use_fp16 = vggt_cfg.get('cache_use_fp16', False)  # Store as float16
        self.cache_compress = vggt_cfg.get('cache_compress', False)  # Use compression
        
        if self.use_vggt_cache:
            from collections import OrderedDict
            self.vggt_feature_cache = OrderedDict()  # In-memory LRU cache: (sample_idx, cam_idx) -> features
            
            # Create cache directory if it doesn't exist
            import os
            os.makedirs(self.vggt_cache_dir, exist_ok=True)
            
            # Load existing cache from disk if available
            self._load_cache_from_disk()
            
            print(f"[VGGT Distillation] Feature caching enabled")
            print(f"[VGGT Distillation] Cache directory: {self.vggt_cache_dir}")
            print(f"[VGGT Distillation] Loaded {len(self.vggt_feature_cache)} cached samples from disk")
            print(f"[VGGT Distillation] Max cache size: {self.vggt_cache_max_size}")
            if self.cache_selected_only:
                print(f"[VGGT Distillation] Cache mode: Selected layers only {self.vggt_layer_indices}")
            else:
                print(f"[VGGT Distillation] Cache mode: All 24 layers (flexible)")
            if self.cache_use_fp16:
                print(f"[VGGT Distillation] Using float16 storage (50% space savings)")
            if self.cache_compress:
                print(f"[VGGT Distillation] Using compression (~40% space savings)")
        
        # Get student channels from img_backbone output
        # For ResNet50: [512, 1024, 2048] before FPN
        # After FPN with out_channels (hidden_dim): all become same dimension
        student_channels = vggt_cfg.get('student_channels', [512, 1024, 2048])
        
        print(f"[VGGT Distillation] Using layers: {self.vggt_layer_indices}")
        print(f"[VGGT Distillation] Student channels: {student_channels}")
        
        # Build distillation loss
        distill_loss_cfg = vggt_cfg.get('distill_loss', {})
        distill_loss_cfg.update({
            'teacher_channels': self.vggt_teacher_channels,
            'student_channels_list': student_channels
        })
        
        self.distillation_loss = build_loss({
            'type': 'FeatureDistillationLoss',
            **distill_loss_cfg
        })
        
        # Move distillation loss to the same device as the model
        # This is necessary when _init_vggt_distillation is called lazily during training
        # (after the model has already been moved to GPU)
        device = next(self.parameters()).device
        self.distillation_loss = self.distillation_loss.to(device)
        
        print(f"[VGGT Distillation] Distillation loss configured with weight: {distill_loss_cfg.get('loss_weight', 1.0)}")
    
    def _ensure_vggt_feature_distillation_initialized(self):
        """Lazy initialization of VGGT feature distillation components."""
        if not self.vggt_feature_initialized and self.use_vggt_feature_distillation:
            self._init_vggt_feature_distillation(self.vggt_feature_distillation_config)
            self.vggt_feature_initialized = True
        elif self.use_vggt_feature_distillation:
            self._ensure_vggt_feature_cache_ready()
    
    def _ensure_vggt_feature_cache_ready(self):
        if self.use_vggt_cache and not self.vggt_feature_cache_ready:
            print(f"[VGGT Feature Distillation] Cache enabled: {self.vggt_cache_dir}")
            print(f"[VGGT Feature Distillation] Cache max size: {self.vggt_cache_max_size}")
            print(f"[VGGT Feature Distillation] Cache selected layers only: {self.cache_selected_only}")
            print(f"[VGGT Feature Distillation] Cache FP16: {self.cache_use_fp16}")
            print(f"[VGGT Feature Distillation] Cache compress: {self.cache_compress}")
            self._load_cache_from_disk()
            self.vggt_feature_cache_ready = True

    def _init_vggt_feature_distillation(self, config, load_cache=True):
        """Initialize VGGT feature distillation components.
        
        This sets up:
        1. Feature head (per-Gaussian feature vectors)
        2. Feature projector (VGGT features -> compact space)
        3. Feature distillation loss
        4. Optional temporal contrastive loss
        """
        if config is None:
            config = {}
        
        print("[VGGT Feature Distillation] Initializing components...")
        
        if load_cache:
            self._ensure_vggt_feature_cache_ready()
        
        # Get configuration
        gaussian_feature_dim = config.get('gaussian_feature_dim', 32)
        teacher_dim = self.vggt_teacher_channels  # VGGT outputs 2048-dim features
        num_head_layers = config.get('num_head_layers', 2)
        
        # Create feature head (per-Gaussian feature vector)
        feature_layers = [nn.Linear(self.in_channels, self.in_channels*2), nn.LeakyReLU(inplace=True)]
        for i in range(num_head_layers - 1):
            feature_layers.extend([
                nn.Linear(self.in_channels*2, self.in_channels*2),
                nn.LeakyReLU(inplace=True)
            ])
        feature_layers.append(nn.Linear(self.in_channels*2, gaussian_feature_dim))
        self.feature_head = nn.Sequential(*feature_layers)
        
        # Create feature projector (VGGT 2048-dim -> compact space)
        self.feature_projector = FeatureProjector(
            teacher_dim=teacher_dim,
            gaussian_feature_dim=gaussian_feature_dim,
            **config.get('projector', {})
        )
        
        # Create feature distillation loss
        self.feature_distillation_loss = FeatureDistillationLoss(
            **config.get('distillation_loss', {})
        )
        
        # Optional: temporal contrastive loss
        if config.get('use_temporal_contrast', False):
            self.temporal_feature_loss = TemporalFeatureContrastiveLoss(
                **config.get('temporal_loss', {})
            )
        
        # Move to device
        device = next(self.parameters()).device
        self.feature_head = self.feature_head.to(device)
        self.feature_projector = self.feature_projector.to(device)
        self.feature_distillation_loss = self.feature_distillation_loss.to(device)
        if self.temporal_feature_loss is not None:
            self.temporal_feature_loss = self.temporal_feature_loss.to(device)
        
        print(f"[VGGT Feature Distillation] Using VGGT layers: {self.vggt_layer_indices}")
        print(f"[VGGT Feature Distillation] Teacher dim: {teacher_dim}")
        print(f"[VGGT Feature Distillation] Gaussian feature dim: {gaussian_feature_dim}")
        print(f"[VGGT Feature Distillation] ✅ All components initialized and moved to {device}")
    
    def _load_cache_from_disk(self):
        """Verify cache directory exists for on-demand loading.
        
        With pure on-demand loading, we don't pre-load any cache at startup.
        Features are loaded from disk only when needed during training.
        
        Benefits:
        - Instant training startup (no waiting for cache loading)
        - Zero memory usage until features are needed
        - LRU-style caching (frequently used features stay in memory)
        
        Supports optimized cache formats:
        - Compressed files (.pt.gz)
        - Float16 storage
        - Selected layers only
        """
        if not os.path.exists(self.vggt_cache_dir):
            print(f"[VGGT Cache] Warning: Cache directory not found: {self.vggt_cache_dir}")
            print(f"[VGGT Cache] Features will be computed online (slow!)")
            return
        
        # Count available cache files for reporting
        import glob
        cache_files = glob.glob(os.path.join(self.vggt_cache_dir, 'sample_*.pt'))
        cache_files_gz = glob.glob(os.path.join(self.vggt_cache_dir, 'sample_*.pt.gz'))
        total_cache_files = len(cache_files) + len(cache_files_gz)
        
        print(f"[VGGT Cache] On-demand loading enabled")
        print(f"[VGGT Cache] Found {total_cache_files} cache files on disk")
        print(f"[VGGT Cache] In-memory cache capacity: {self.vggt_cache_max_size} files")
    
    def _add_to_lru_cache(self, cache_key, features):
        """Add features to LRU cache, evicting oldest if full.
        
        Args:
            cache_key (tuple): (sample_idx, cam_idx)
            features (list[Tensor]): List of feature tensors
        """
        # Add to cache
        self.vggt_feature_cache[cache_key] = features
        
        # If cache exceeds max size, remove oldest (first) item
        if len(self.vggt_feature_cache) > self.vggt_cache_max_size:
            # popitem(last=False) removes the oldest (first) item
            self.vggt_feature_cache.popitem(last=False)
    
    def _load_single_cache_from_disk(self, sample_idx, cam_idx):
        """Load a single cache file from disk on-demand.
        
        Args:
            sample_idx (str): Sample token (UUID string)
            cam_idx (int): Camera index
            
        Returns:
            list[Tensor] or None: Cached features, or None if not found
        """
        # Determine cache file path
        if self.cache_compress:
            cache_file = os.path.join(self.vggt_cache_dir, f'sample_{sample_idx}_cam_{cam_idx}.pt.gz')
            if not os.path.exists(cache_file):
                cache_file = os.path.join(self.vggt_cache_dir, f'sample_{sample_idx}_cam_{cam_idx}.pt')
        else:
            cache_file = os.path.join(self.vggt_cache_dir, f'sample_{sample_idx}_cam_{cam_idx}.pt')
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Load from disk
            if cache_file.endswith('.gz'):
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    data = torch.load(f, map_location='cpu')
            else:
                data = torch.load(cache_file, map_location='cpu')
            
            # Handle both old format (list) and new format (dict)
            if isinstance(data, dict):
                features = data['features']
            else:
                # Old format: just features
                features = data
            
            # Convert from float16 if needed
            if self.cache_use_fp16:
                features = [f.float() for f in features]
            
            return features
            
        except Exception as e:
            print(f"[VGGT Cache] Warning: Failed to load {cache_file}: {e}")
            return None
    
    def _save_cache_to_disk(self, cache_key, features):
        """Save cached features to disk for reuse in future training runs.
        
        Args:
            cache_key (tuple): (sample_idx, cam_idx)
            features (list[Tensor]): List of feature tensors for each layer
        """
        
        sample_idx, cam_idx = cache_key
        
        # Determine file extension based on compression
        if self.cache_compress:
            cache_file = os.path.join(
                self.vggt_cache_dir,
                f'sample_{sample_idx}_cam_{cam_idx}.pt.gz'
            )
        else:
            cache_file = os.path.join(
                self.vggt_cache_dir,
                f'sample_{sample_idx}_cam_{cam_idx}.pt'
            )
        
        # Skip if already exists on disk
        if os.path.exists(cache_file):
            return
        
        try:
            # Convert to float16 if requested
            if self.cache_use_fp16:
                features = [f.half() for f in features]
            
            # Save to disk (features already on CPU from _store_cached_vggt_features)
            if self.cache_compress:
                import gzip
                import io
                
                # Save to memory buffer first
                buffer = io.BytesIO()
                torch.save(features, buffer)
                buffer.seek(0)
                
                # Compress and write to disk
                with gzip.open(cache_file, 'wb', compresslevel=6) as f:
                    f.write(buffer.read())
            else:
                torch.save(features, cache_file)
                
        except Exception as e:
            print(f"[VGGT Cache] Warning: Failed to save {cache_file}: {e}")

    def extract_vggt_teacher_features(self, img, num_cams=6, img_metas=None):
        """Extract features from VGGT teacher model with caching support.
        
        CRITICAL OPTIMIZATION: During training, images are augmented differently each iteration
        (resize, crop, flip, rotate). However, VGGT is very slow (~18s per forward pass).
        
        Solution: Cache features at ORIGINAL image resolution BEFORE augmentation.
        When augmented images arrive, we load cached features and apply spatial transformations
        to match the augmentation. This is MUCH faster than recomputing VGGT.
        
        Args:
            img (Tensor): Input images [B*N, C, H, W] where N is number of cameras
            num_cams (int): Number of camera views (default: 6 for nuScenes)
            img_metas (list[dict]): Metadata with sample identifiers for caching
            
        Returns:
            list[Tensor]: VGGT features for selected layers, each [B*N, C, H', W']
        """
        from .vggt_utils import extract_vggt_features
        
        B_N, C, orig_height, orig_width = img.size()
        B = B_N // num_cams  # Recover batch size
        N = num_cams
        
        # Check if we can use cached features
        if self.use_vggt_cache and img_metas is not None and self.training:
            # Try to load from cache
            vggt_features = self._load_cached_vggt_features(img, B, N, img_metas)
            if vggt_features is not None:
                return vggt_features
        
        # Cache miss or disabled - compute features
        # CRITICAL: Only load VGGT model now if we actually need it (cache miss)
        if self.vggt_model is None:
            print("[VGGT Distillation] Cache miss - loading VGGT model on-demand...")
            self._load_vggt_model()
        
        # Resize to multiple of patch_size
        new_height = self.patch_size * round(orig_height / self.patch_size)
        new_width = self.patch_size * round(orig_width / self.patch_size)
        
        # Reshape to [B*N, C, H, W] for batch interpolation
        vggt_img = F.interpolate(
            img, 
            size=(new_height, new_width),
            mode='bilinear', 
            align_corners=False
        )
        
        # Reshape to [B, N, C, H, W] for multi-view processing
        vggt_img = vggt_img.view(B, N, C, new_height, new_width)
        
        _, S, _, H, W = vggt_img.shape  # S = N (number of views/sequence length)
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # Extract VGGT features with multi-view aggregation (frozen, no gradients)
        with torch.no_grad():
            aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(vggt_img)
            
            # For caching: extract ALL layers OR only selected layers based on config
            # For loss computation: extract only selected layers
            if self.use_vggt_cache and img_metas is not None and self.training:
                if self.cache_selected_only:
                    # Cache only selected layers (saves ~20-23x disk space!)
                    selected_vggt_features = extract_vggt_features(
                        aggregated_tokens_list, 
                        ps_idx, 
                        self.vggt_layer_indices,  # Only selected layers
                        B, S,
                        patch_h, patch_w
                    )
                    
                    # Store selected layers in cache
                    self._store_cached_vggt_features(selected_vggt_features, B, N, img_metas)
                    
                    # Use selected layers directly for loss
                    vggt_features = selected_vggt_features
                    
                else:
                    # Cache all 24 layers for maximum flexibility
                    all_layer_indices = list(range(len(aggregated_tokens_list)))
                    all_vggt_features = extract_vggt_features(
                        aggregated_tokens_list, 
                        ps_idx, 
                        all_layer_indices,
                        B, S,
                        patch_h, patch_w
                    )
                    
                    # Store ALL layers in cache
                    self._store_cached_vggt_features(all_vggt_features, B, N, img_metas)
                    
                    # Extract selected layers for loss computation
                    vggt_features = [all_vggt_features[i] for i in self.vggt_layer_indices]
                    
            else:
                # Only extract selected layers (when caching disabled)
                vggt_features = extract_vggt_features(
                    aggregated_tokens_list, 
                    ps_idx, 
                    self.vggt_layer_indices, 
                    B, S,
                    patch_h, patch_w
                )
        
        return vggt_features
    
    def _load_cached_vggt_features(self, img, B, N, img_metas):
        """Load VGGT features from cache if available.
        
        Cache may contain ALL 24 layers (flexible mode) or only selected layers (space-saving mode).
        Returns features for selected layers for loss computation.
        
        Returns None if cache miss, otherwise returns cached features for selected layers.
        """
        # Special case: cache_max_size=0 means no in-memory cache, always load from disk
        if self.vggt_cache_max_size == 0:
            return self._load_cached_features_no_memory(img, B, N, img_metas)
        
        # Check if all samples are in cache (memory or disk)
        all_cached = True
        for b in range(B):
            # Use sample_idx as unique identifier (independent of augmentation)
            sample_idx = img_metas[b].get('sample_idx', None)
            if sample_idx is None:
                all_cached = False
                break
            
            # Check all cameras for this sample
            for cam_idx in range(N):
                cache_key = (sample_idx, cam_idx)
                
                # If not in memory, try loading from disk on-demand
                if cache_key not in self.vggt_feature_cache:
                    features = self._load_single_cache_from_disk(sample_idx, cam_idx)
                    
                    if features is not None:
                        # Successfully loaded - add to LRU cache
                        self._add_to_lru_cache(cache_key, features)
                    else:
                        # Cache file doesn't exist or failed to load
                        all_cached = False
                        break
                else:
                    # Move to end (mark as recently used)
                    self.vggt_feature_cache.move_to_end(cache_key)
            
            if not all_cached:
                break
        
        if not all_cached:
            return None
        
        # All features are cached - load and return selected ones
        # Calculate target feature size based on current image size
        B_N, C_img, H_img, W_img = img.shape
        patch_h = round(H_img / self.patch_size)
        patch_w = round(W_img / self.patch_size)
        
        # Get first cached sample to check format
        first_key = (img_metas[0]['sample_idx'], 0)
        first_cached = self.vggt_feature_cache[first_key]
        num_cached_layers = len(first_cached)
        
        if self.cache_selected_only:
            # Cache contains only selected layers - use directly
            # Layers are in the same order as self.vggt_layer_indices
            cached_features_list = [[] for _ in range(num_cached_layers)]
            
            # CRITICAL: Loop order ensures correct sample-camera matching
            # Features are appended in order: batch0_cam0, batch0_cam1, ..., batch0_cam5,
            #                                  batch1_cam0, batch1_cam1, ..., batch1_cam5, ...
            # This matches the order of imgs_flat: [B*N, C, H, W]
            for b in range(B):
                sample_idx = img_metas[b]['sample_idx']  # Unique sample identifier
                for cam_idx in range(N):
                    cache_key = (sample_idx, cam_idx)  # Cache key: (sample, camera)
                    cached_feats = self.vggt_feature_cache[cache_key]
                    
                    for layer_i, feat in enumerate(cached_feats):
                        feat = feat.to(img.device)
                        _, _, H_feat, W_feat = feat.shape
                        
                        if H_feat != patch_h or W_feat != patch_w:
                            feat = F.interpolate(
                                feat, 
                                size=(patch_h, patch_w),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        # Append in ORDER: guarantees cached_features_list matches imgs_flat order
                        cached_features_list[layer_i].append(feat.squeeze(0))
            
            vggt_features = [torch.stack(feats) for feats in cached_features_list]
            return vggt_features
            
        else:
            # Cache contains all layers - extract selected ones
            all_cached_features_list = [[] for _ in range(num_cached_layers)]
            
            for b in range(B):
                sample_idx = img_metas[b]['sample_idx']
                for cam_idx in range(N):
                    cache_key = (sample_idx, cam_idx)
                    cached_feats = self.vggt_feature_cache[cache_key]
                    
                    for layer_i, feat in enumerate(cached_feats):
                        feat = feat.to(img.device)
                        _, _, H_feat, W_feat = feat.shape
                        
                        if H_feat != patch_h or W_feat != patch_w:
                            feat = F.interpolate(
                                feat, 
                                size=(patch_h, patch_w),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        all_cached_features_list[layer_i].append(feat.squeeze(0))
            
            # Stack all layers
            all_vggt_features = [torch.stack(feats) for feats in all_cached_features_list]
            
            # Return only the selected layers for loss computation
            selected_vggt_features = [all_vggt_features[i] for i in self.vggt_layer_indices]
            return selected_vggt_features
    
    def _load_cached_features_no_memory(self, img, B, N, img_metas):
        """Load cached features directly from disk without in-memory cache.
        
        This is used when cache_max_size=0 to minimize memory usage.
        Features are loaded from disk on every access.
        
        Args:
            img (Tensor): Input images [B*N, C, H, W]
            B (int): Batch size
            N (int): Number of cameras
            img_metas (list[dict]): Image metadata with sample_idx
            
        Returns:
            list[Tensor] or None: List of feature tensors for each layer, or None if cache missing
        """
        # Calculate target feature size
        B_N, C_img, H_img, W_img = img.shape
        patch_h = round(H_img / self.patch_size)
        patch_w = round(W_img / self.patch_size)
        
        # Check all samples exist and load first to determine format
        sample_idx_0 = img_metas[0].get('sample_idx', None)
        if sample_idx_0 is None:
            return None
        
        first_cached = self._load_single_cache_from_disk(sample_idx_0, 0)
        if first_cached is None:
            return None
        
        num_cached_layers = len(first_cached)
        
        # Load all features directly from disk
        if self.cache_selected_only:
            cached_features_list = [[] for _ in range(num_cached_layers)]
            
            for b in range(B):
                sample_idx = img_metas[b].get('sample_idx', None)
                if sample_idx is None:
                    return None
                
                for cam_idx in range(N):
                    # Load from disk each time (no in-memory storage)
                    cached_feats = self._load_single_cache_from_disk(sample_idx, cam_idx)
                    
                    if cached_feats is None:
                        return None
                    
                    for layer_i, feat in enumerate(cached_feats):
                        feat = feat.to(img.device)
                        _, _, H_feat, W_feat = feat.shape
                        
                        if H_feat != patch_h or W_feat != patch_w:
                            feat = F.interpolate(
                                feat,
                                size=(patch_h, patch_w),
                                mode='bilinear',
                                align_corners=False
                            )
                        
                        cached_features_list[layer_i].append(feat.squeeze(0))
            
            # Stack all features
            vggt_features = [torch.stack(feats) for feats in cached_features_list]
            return vggt_features
        
        return None
    
    def _store_cached_vggt_features(self, vggt_features, B, N, img_metas):
        """Store VGGT features in memory cache and save to disk.
        
        We cache features per (sample_idx, camera) pair at the current resolution.
        Since augmentation varies per iteration, cached features will be spatially
        transformed to match new augmentations.
        
        Features are saved to disk for persistence across training runs.
        """
        # Check cache size limit
        if len(self.vggt_feature_cache) >= self.vggt_cache_max_size:
            # Cache is full - don't store
            return
        
        # Store features for each sample and camera
        for b in range(B):
            sample_idx = img_metas[b].get('sample_idx', None)
            if sample_idx is None:
                continue
            
            for cam_idx in range(N):
                cache_key = (sample_idx, cam_idx)
                
                # Skip if already cached
                if cache_key in self.vggt_feature_cache:
                    continue
                
                # Extract features for this specific camera
                feats_to_cache = []
                for layer_feats in vggt_features:
                    # layer_feats: [B*N, C, H, W]
                    idx = b * N + cam_idx
                    feats_to_cache.append(layer_feats[idx:idx+1].cpu())  # Move to CPU to save GPU memory
                
                # Store in memory cache
                self.vggt_feature_cache[cache_key] = feats_to_cache
                
                # Save to disk for persistence
                self._save_cache_to_disk(cache_key, feats_to_cache)

    def extract_img_feat(self, img, img_metas):
        """Extract image features with optional intermediate features for distillation.
        
        Overrides parent method to also return backbone features before FPN
        for distillation purposes.
        
        Args:
            img (Tensor): Input images
            img_metas (list[dict]): Image metadata
            
        Returns:
            tuple: (final_features, intermediate_features) where intermediate_features
                   is None during inference or when distillation is disabled
        """
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            
            # Extract backbone features
            img_feats = self.img_backbone(img)
            
            # Store intermediate features for distillation (before FPN)
            intermediate_feats = None
            if self.use_vggt_distillation and self.training:
                # Ensure VGGT is initialized before accessing distillation_loss
                self._ensure_vggt_initialized()
                
                # img_feats is a tuple of (C3, C4, C5) for ResNet with out_indices=(1, 2, 3)
                # For speed optimization, only extract the levels we need
                num_student_layers = len(self.distillation_loss.teacher_projs)
                
                if num_student_layers == 1:
                    # Single layer mode (fastest): extract only C4 (index 1)
                    # C4 has 1024 channels at OS=16 (good balance of speed and semantics)
                    intermediate_feats = [img_feats[1]]  # Only C4
                else:
                    # Full mode: extract all levels C3, C4, C5
                    intermediate_feats = list(img_feats)
        else:
            return None, None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        # During training with distillation, return both final and intermediate features
        if self.use_vggt_distillation and self.training:
            return img_feats, intermediate_feats
        else:
            return img_feats, None

    def create_voxel_centers(self, grid_cfg):
        Z = int((grid_cfg['z'][1] - grid_cfg['z'][0]) / grid_cfg['z'][2])
        H = int((grid_cfg['x'][1] - grid_cfg['x'][0]) / grid_cfg['x'][2])
        W = int((grid_cfg['y'][1] - grid_cfg['y'][0]) / grid_cfg['y'][2])
        self.Z, self.H, self.W = Z, H, W

        xs = torch.linspace(0.5 * grid_cfg['x'][2] + grid_cfg['x'][0], grid_cfg['x'][1] - 0.5 * grid_cfg['x'][2], W).view(W, 1, 1).expand(W, H, Z)
        ys = torch.linspace(0.5 * grid_cfg['y'][2] + grid_cfg['y'][0], grid_cfg['y'][1] - 0.5 * grid_cfg['y'][2], H).view(1, H, 1).expand(W, H, Z)
        zs = torch.linspace(0.5 * grid_cfg['z'][2] + grid_cfg['z'][0], grid_cfg['z'][1] - 0.5 * grid_cfg['z'][2], Z).view(1, 1, Z).expand(W, H, Z)
        
        ref_3d = torch.stack((xs, ys, zs), -1)
        return ref_3d
    
    def create_voxel_centers_random(self, grid_cfg, N):

        x = torch.rand(N).uniform_(grid_cfg['x'][0], grid_cfg['x'][1])
        y = torch.rand(N).uniform_(grid_cfg['y'][0], grid_cfg['y'][1])
        z = torch.rand(N).uniform_(grid_cfg['z'][0], grid_cfg['z'][1])
        
        points = torch.vstack((x, y, z)).T
        return points
    
    def init_weights(self):
        super().init_weights()
        # initialize heads
        for m in self.semantic_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=.01)
                nn.init.constant_(m.bias, 0)
        if self.render_rgb:
            for m in self.rgb_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.opacity_head is not None:
            for m in self.opacity_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.scale_head is not None:
            for m in self.scale_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)
        if self.rotation_head is not None:
            for m in self.rotation_head:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=.01)
                    nn.init.constant_(m.bias, 0)

    def gaussians_to_occupancy(self, means, quats, scale, opacity, feature,):
        # Voxelize Gaussians 
        if self.voxel_centers is None:
            self.voxel_centers = self.create_voxel_centers(self.voxel_grid_cfg).to(means.device)
            self.min_positions = torch.tensor([self.voxel_grid_cfg['x'][0], self.voxel_grid_cfg['y'][0], self.voxel_grid_cfg['z'][0]], device=means.device)
        _, inv_covariance = quat_scale_to_covar_preci(quats, scale * self.scale_multiplier)

        mean_vox_index = torch.floor((means - self.min_positions) / self.voxel_grid_cfg['x'][2])
        neighborhood = torch.arange(-self.max_neighborhood, self.max_neighborhood+1, device=means.device)
        neighborhood = torch.stack(torch.meshgrid(neighborhood, neighborhood, neighborhood, indexing='ij'), dim=-1).reshape(-1, 3)
        neighborhood_index = (mean_vox_index[:, None, :] + neighborhood[None, ...]).long()
        neighborhood_index_mask = ((neighborhood_index >= 0) & (neighborhood_index < torch.tensor([self.W, self.H, self.Z], device=means.device))).all(dim=-1)
        neighborhood_index_flat = neighborhood_index[neighborhood_index_mask]
        neighborhood_coords = self.voxel_centers[tuple(neighborhood_index_flat.T)]
        diff = neighborhood_coords - means[:, None, :].repeat(1, neighborhood.shape[0], 1)[neighborhood_index_mask]
        cov_batched = inv_covariance[:, None, ...].repeat(1, neighborhood.shape[0], 1, 1)[neighborhood_index_mask]
        contribs = torch.exp(-0.5 * ((diff[..., None, :] @ cov_batched) @ diff[..., :, None])).squeeze()

        # distribute opacity and features to each voxel
        opacities = opacity[:, None, 0].repeat(1, neighborhood.shape[0])[neighborhood_index_mask] * contribs
        features = feature[:, None, :].repeat(1, neighborhood.shape[0], 1)[neighborhood_index_mask] * contribs[..., None]
        voxel_occupancy = torch.zeros((self.W, self.H, self.Z), device=means.device)
        voxel_semantics = torch.zeros((self.W, self.H, self.Z, self.num_classes), device=means.device)
        indices_unique, labels = neighborhood_index_flat.unique(dim=0, return_inverse=True)
        grouped_opacities = torch.zeros((indices_unique.size(0),), device=labels.device).scatter_add(0, labels, opacities)
        grouped_semantics = torch.zeros((indices_unique.size(0), self.num_classes), device=labels.device).scatter_add(0, labels[:, None].expand_as(features), features)
        voxel_occupancy[tuple(indices_unique.T)] = grouped_opacities.clamp(0, 1)
        voxel_semantics[tuple(indices_unique.T)] = grouped_semantics
        return voxel_occupancy, voxel_semantics

    def ego_motion_compensation(self, ego2global_next):
        prev_means, prev_feat = self.prev_feat[..., :3], self.prev_feat[..., 3:]
        # Apply gaussian flow first
        if self.move_dynamic_gaussians:
            if hasattr(self.gaussian_decoder, 'deformation_module'):
                # New deformation-based approach
                if self.next_t_index is not None:
                    target_timesteps = torch.tensor([[self.next_t_index]], device=prev_means.device, dtype=torch.float32).repeat(ego2global_next.shape[0], 1)
                    deformed_gaussians = self.gaussian_decoder.deformation_module(
                        gaussians_means=prev_means,
                        gaussians_features=prev_feat,
                        time_stamps=target_timesteps
                    )
                    prev_means = deformed_gaussians['means'][:, :, 0, :]  # Take first (and only) timestep
            elif self.temporal_module is not None:
                # Original approach (fallback when no deformation module)
                offsets = self.temporal_module(prev_feat, [torch.tensor([self.next_t_index]).numpy() for i in range(ego2global_next.shape[0])]) # [B, 10000, T, 3]
                prev_means = move_gaussians_temporal_module(prev_means, self.semantic_head(prev_feat), offsets, self.dynamic_classes)[:, 0]
        
        # Then transform to next frame
        cur2next = torch.inverse(ego2global_next) @ self.prev_ego2global
        prev_means = (cur2next[:, None, ...] @ torch.cat((prev_means, prev_means.new_ones(*prev_means.shape[:2], 1)), dim=-1)[..., None])[..., :3, 0]
        self.prev_feat = torch.cat([prev_means, prev_feat], dim=-1)

    def forward_test(self, img_metas, img_inputs=None, **kwargs):
        num_augs = len(img_metas)
        if num_augs == 1:
            return self.simple_test(img_metas[0], img_inputs=img_inputs[0], **kwargs)

    def simple_test(self,
                    img_metas,
                    img_inputs=None,
                    render_preds=False,
                    clip_low_density_regions=True,
                    gs_intrins=None,
                    gs_extrins=None,
                    return_means=False,
                    **kwargs):        
        # Unwrap DataContainer if needed (happens during validation)
        from mmcv.parallel import DataContainer as DC
        if isinstance(img_metas, DC):
            img_metas = img_metas.data[0]
        if isinstance(img_inputs, DC):
            img_inputs = img_inputs.data[0]

        # Reset stored self.prev_feat when the sample is the first sample of a scene
        if not img_metas[0]['has_prev_sample']:
            gaussian_outputs = self.forward_gaussian(img_inputs, img_metas, None)
            prev_means, prev_feature = gaussian_outputs[0], gaussian_outputs[1]
            self.prev_feat = torch.cat([prev_means, prev_feature], dim=-1)
            self.prev_ego2global = img_inputs[3]
        # Apply ego-motion compensation
        self.ego_motion_compensation(img_inputs[3])
        out_dict = {}
        if self.gaussian_decoder.store_intermediate:
            # TODO: Needs major rework
            gaussians_per_block = self.forward_gaussian(img_inputs, img_metas, self.prev_feat)
            means = torch.stack(([g[0] for g in gaussians_per_block]))
            quats = torch.stack(([g[1] for g in gaussians_per_block]))
            scale = torch.stack(([g[2] for g in gaussians_per_block]))
            opacity = torch.stack(([g[3] for g in gaussians_per_block]))
            feature = torch.stack(([g[4] for g in gaussians_per_block]))
            # velocity = torch.stack(([g[5] for g in gaussians_per_block]))
            self.prev_feat = torch.cat([means[0], quats[0], scale[0], opacity[0], feature[0]], dim=-1)
            # self.prev_feat = torch.cat([means[0], quats[0], scale[0], opacity[0], velocity[0], feature[0]], dim=-1) if velocity[0].abs().sum()>1e-2 else None
            sem_feature = self.semantic_head(feature)
            occupancy = [self.gaussians_to_occupancy(means[i][0], quats[i][0], scale[i][0], opacity[i][0], sem_feature[i][0]) for i in range(len(means))]
            density = torch.stack(([o[0] for o in occupancy])) 
            semantics = torch.stack(([o[1] for o in occupancy]))
            if clip_low_density_regions:
                density[density<1e-3] = 0
                density[..., 11:] = 0
            
            out_dict['previous_density'] = density[:-1]
            out_dict['previous_occ'] = occupancy[:-1]
            out_dict['previous_means'] = means[:-1]
            
        else:
            gaussian_outputs = self.forward_gaussian(img_inputs, img_metas, self.prev_feat)
            means, feature = gaussian_outputs[0], gaussian_outputs[1]
            self.prev_feat = torch.cat([means, feature], dim=-1)
            self.prev_ego2global = img_inputs[3]
            
            # First, generate ALL canonical properties, just like in forward_train
            sem_feature = self.semantic_head(feature)
            opacity = self.opacity_head(feature) if self.opacity_head is not None else torch.ones_like(means[..., :1])
            scale = self.scale_head(feature) if self.scale_head is not None else torch.full_like(means, .3)
            if self.scale_range is not None:
                scale = self.scale_range[0] + scale * (self.scale_range[1] - self.scale_range[0])
            quats = F.normalize(self.rotation_head(feature), dim=-1) if self.rotation_head is not None else torch.tensor([1., 0., 0., 0.], device=means.device).repeat(means.shape[0], means.shape[1], 1)
           
            # Voxelize using canonical properties
            density, semantics = self.gaussians_to_occupancy(means[0], quats[0], scale[0], opacity[0], sem_feature[0])

        # clip low density regions & remove roof
        if clip_low_density_regions:
            density[density<1e-3] = 0
            density[..., 11:] = 0

        if render_preds:
            if type(gs_intrins) == list:
                gs_intrins = gs_intrins[0]
                gs_extrins = gs_extrins[0]
            rendered_outs_no_temporal = self.rasterizer(means, quats, scale, opacity, sem_feature, gs_intrins, gs_extrins, mode='RGB+D')
            out_dict['rendered_semantics'] = (rendered_outs_no_temporal[..., :-1].argmax(dim=-1) + 
                                              int(not self.with_others)).squeeze(0).to(torch.uint8).cpu().numpy()
            out_dict['rendered_depths'] = rendered_outs_no_temporal[..., -1].squeeze(0).cpu().numpy()
            
        # combine density and semantics
        semantics = semantics.argmax(dim=-1) + int(not self.with_others)
        free_space = torch.stack([density < tr for tr in self.eval_threshold_range])

        out_dict['occupancy'] = semantics.to(torch.uint8).cpu().numpy()
        out_dict['free_space'] = free_space.cpu().numpy()

        if return_means:
            out_dict['means'] = means.squeeze(0).cpu().numpy()
            out_dict['opacity'] = opacity.squeeze(0).cpu().numpy()
            out_dict['feature'] = sem_feature.squeeze(0).cpu().numpy()
            out_dict['label'] = (sem_feature.argmax(dim=-1) + int(not self.with_others)).squeeze(0).cpu().numpy()
            out_dict['scale'] = scale.squeeze(0).cpu().numpy()
            out_dict['quats'] = quats.squeeze(0).cpu().numpy()

        return [out_dict]

    def prepare_prev_feat(self, img_inputs, img_metas):
        with torch.no_grad():
            prev_feat = None
            T = img_inputs[0].shape[1]
            for t in reversed(range(1, T)): # dont compute for current
                img_inputs_cur = [i[:, t, ...].contiguous() for i in img_inputs[:-1]] + [img_inputs[-1]]
                # forward_gaussian returns different number of values based on training mode and distillation
                gaussian_outputs = self.forward_gaussian(img_inputs_cur, img_metas, prev_feat)
                # Unpack: could be (means, feature) or (means, feature, vggt_feats, student_feats)
                means, feature = gaussian_outputs[0], gaussian_outputs[1]
                # Ego motion compensation
                ego2global_c = img_inputs_cur[3]
                ego2global_next = img_inputs[3][:, t-1]

                # Apply gaussian flow first
                # means [B, N, 3], feature [B, N, C]
                if self.move_dynamic_gaussians: # default: False; Ours: True
                    if hasattr(self.gaussian_decoder, 'deformation_module'):
                        # New deformation-based approach
                        target_timesteps = torch.tensor([[self.next_t_index]], device=means.device, dtype=torch.float32).repeat(len(img_metas), 1)
                        deformed_gaussians = self.gaussian_decoder.deformation_module(
                            gaussians_means=means,
                            gaussians_features=feature,
                            time_stamps=target_timesteps
                        )
                        means = deformed_gaussians['means'][:, :, 0, :]  # Take first (and only) timestep
                    elif self.temporal_module is not None:
                        # Original approach (fallback when no deformation module)
                        target_timesteps = [np.array([self.next_t_index]) for i in range(len(img_metas))]
                        offsets = self.temporal_module(feature, target_timesteps)
                        means = move_gaussians_temporal_module(means, self.semantic_head(feature), offsets, self.dynamic_classes)[:, 0]

                # Then transform to next frame
                cur2next = torch.inverse(ego2global_next) @ ego2global_c
                means = (cur2next[:, None, ...] @ torch.cat((means, means.new_ones(*means.shape[:2], 1)), dim=-1)[..., None])[..., :3, 0]
                prev_feat = torch.cat([means, feature], dim=-1)

        return prev_feat

    def forward_gaussian(self, img_inputs, img_metas, prev_feat=None):            
        # Gaussian Init
        input_shape = torch.tensor(img_inputs[0].shape[-2:])
        B = img_inputs[1].shape[0]
        means = self.initial_means[None, ...].repeat(B, 1, 1)
        feature = self.gaussian_queries[None, ...].repeat(B, 1, 1)

        # Extract VGGT teacher features if distillation is enabled and training
        vggt_feats = None
        if self.use_vggt_distillation and self.training:
            # Lazy initialization: only load VGGT on first training batch
            self._ensure_vggt_initialized()
            
            # img_inputs[0] is [B, N, C, H, W], need to flatten to [B*N, C, H, W]
            imgs = img_inputs[0]
            if imgs.dim() == 5:
                B_img, N, C, H, W = imgs.shape
                imgs_flat = imgs.view(B_img * N, C, H, W)
                num_cams = N
            else:
                imgs_flat = imgs
                num_cams = 1
            # Multi-view VGGT with caching support: processes all N cameras together
            # Caching dramatically speeds up training by reusing features across iterations
            vggt_feats = self.extract_vggt_teacher_features(imgs_flat, num_cams=num_cams, img_metas=img_metas)

        # Extract img feats (returns intermediate features during training with distillation)
        img_feats, student_feats = self.extract_img_feat(img_inputs[0], img_metas) # [B*N, C, H_f, W_f]

        # Gaussian Transformer - pass img_metas for temporal deformation only if it's the new TemporalGaussianDecoder
        if hasattr(self.gaussian_decoder, 'deformation_module'):
            # New TemporalGaussianDecoder that supports img_metas
            means, feature = self.gaussian_decoder(means, feature, prev_feat, img_feats, img_inputs[4:8], input_shape, img_metas)
        else:
            # Original gaussian decoder that doesn't accept img_metas
            means, feature = self.gaussian_decoder(means, feature, prev_feat, img_feats, img_inputs[4:8], input_shape)       
        
        # Return features for distillation during training
        if self.use_vggt_distillation and self.training:
            return means, feature, vggt_feats, student_feats
        else:
            return means, feature

    def forward_train(self,
                      img_metas=None,
                      img_inputs=None,
                      gs_gts=None,
                      gs_gts_pixel=None,
                      gs_intrins=None,
                      gs_extrins=None,
                      voxel_semantics=None,
                      mask_camera=None,
                      **kwargs):
        # img inputs with temporal:
        # img: [B, T, N, C, H, W], sensor2ego: [B, T, N, 4, 4], ego2global: [B, T, N, 4, 4], ego_l2global: [B, T, 4, 4],
        # cam2ego_l: [B, T, N, 4, 4], intrins: [B, T, N, 3, 3], # post_rot: [B, T, N, 3, 3], post_trans: [B, T, N, 3], bda: [B, 3, 3]
        assert not ((gs_gts is not None) and (gs_gts_pixel is not None)), "Only one of gs_gts or gs_gts_pixel should be provided"
        losses = dict()
        # reset self.prev_feat after eval
        if self.prev_feat is not None:
            self.prev_feat = None

        # Buildup of previous features if using temporal self-attn
        if img_inputs[0].ndim == 6:
            img_inputs_cur = [i[:, 0, ...].contiguous() for i in img_inputs[:-1]] + [img_inputs[-1]]
            prev_feat = self.prepare_prev_feat(img_inputs, img_metas)
        else:
            img_inputs_cur = img_inputs
            prev_feat = None
        # img_inputs_cur is list with [9, B, N, 3, H, W]

        # Forward gaussian with optional distillation features
        gaussian_outputs = self.forward_gaussian(img_inputs_cur, img_metas, prev_feat)
        if self.use_vggt_distillation and self.training:
            means, feature, vggt_feats, student_feats = gaussian_outputs
        else:
            means, feature = gaussian_outputs
            vggt_feats, student_feats = None, None
        
        # Add VGGT distillation loss if enabled
        if self.use_vggt_distillation and vggt_feats is not None and student_feats is not None:
            distill_loss = self.distillation_loss(vggt_feats, student_feats)
            losses['loss_vggt_distill'] = distill_loss
        
        # Estimate gaussian properties
        sem_pred = self.semantic_head(feature)
        opacity = self.opacity_head(feature) if self.opacity_head is not None else torch.ones_like(means[..., :1])
        scale = self.scale_head(feature) if self.scale_head is not None else torch.full_like(means, .3)
        if self.scale_range is not None:
            scale = self.scale_range[0] + scale * (self.scale_range[1] - self.scale_range[0])
        quats = F.normalize(self.rotation_head(feature), dim=-1) if self.rotation_head is not None else torch.tensor([1., 0., 0., 0.], device=means.device).repeat(means.shape[0], means.shape[1], 1)

        # Using temporal module or deformation module
        if hasattr(self.gaussian_decoder, 'deformation_module'):
            # New deformation-based approach using selected_frames like original temporal module
            target_frames_list = [i['selected_frames'][:-1] for i in img_metas]  # Remove current frame, keep past frames
            
            # Convert to proper tensor format [B, T] - similar to original temporal module
            max_T = max(len(frames) for frames in target_frames_list) if target_frames_list and any(len(frames) > 0 for frames in target_frames_list) else 0
            
            if max_T > 0:
                B = len(target_frames_list)
                target_timesteps = torch.zeros((B, max_T), device=means.device, dtype=torch.float32)
                for b, frames in enumerate(target_frames_list):
                    if len(frames) > 0:
                        target_timesteps[b, :len(frames)] = torch.tensor(frames, dtype=torch.float32)

                deformed_gaussians = self.gaussian_decoder.deformation_module(
                    gaussians_means=means,
                    gaussians_features=feature, 
                    time_stamps=target_timesteps,
                    gaussians_rotations=quats,
                    gaussians_scales=scale,
                    gaussians_opacity=opacity
                ) # Returns [B, N, T, 3] for means
                
                # For temporal rendering, we need to handle all timesteps like original temporal module
                # means will be [B, T+1, N, 3] similar to move_gaussians_temporal_module output
                deformed_means = deformed_gaussians['means']  # [B, N, T, 3]
                
                # Combine current frame (canonical) with deformed frames
                # means [B, N, 3] -> [B, N, 1, 3], then concat with deformed_means [B, N, T, 3]
                canonical_means_expanded = means.unsqueeze(-2)  # [B, N, 1, 3]
                all_means = torch.cat([deformed_means, canonical_means_expanded], dim=-2)  # [B, N, T+1, 3]
                means = all_means.permute(0, 2, 1, 3)  # [B, T+1, N, 3] - same format as original temporal module
                
                # Extract other properties if they were deformed
                if 'rotations' in deformed_gaussians:
                    deformed_quats = deformed_gaussians['rotations']  # [B, N, T, 4]
                    canonical_quats_expanded = quats.unsqueeze(-2)  # [B, N, 1, 4]
                    all_quats = torch.cat([deformed_quats, canonical_quats_expanded], dim=-2)  # [B, N, T+1, 4]
                    quats = all_quats.permute(0, 2, 1, 3)  # [B, T+1, N, 4]
                    
                if 'scales' in deformed_gaussians:
                    deformed_scales = deformed_gaussians['scales']  # [B, N, T, 3]
                    canonical_scales_expanded = scale.unsqueeze(-2)  # [B, N, 1, 3]
                    all_scales = torch.cat([deformed_scales, canonical_scales_expanded], dim=-2)  # [B, N, T+1, 3]
                    scale = all_scales.permute(0, 2, 1, 3)  # [B, T+1, N, 3]
                    
                if 'opacity' in deformed_gaussians:
                    deformed_opacity = deformed_gaussians['opacity']  # [B, N, T, 1]
                    canonical_opacity_expanded = opacity.unsqueeze(-2)  # [B, N, 1, 1]
                    all_opacity = torch.cat([deformed_opacity, canonical_opacity_expanded], dim=-2)  # [B, N, T+1, 1]
                    opacity = all_opacity.permute(0, 2, 1, 3)  # [B, T+1, N, 1]
                
                # Add deformation losses with weights from train_cfg
                if self.train_cfg is not None:
                    if 'deformation_reg_loss' in deformed_gaussians:
                        deformation_weight = self.train_cfg.get('deformation_weight', 0.1)
                        losses['loss_deformation'] = deformed_gaussians['deformation_reg_loss'] * deformation_weight
                    
                    if 'static_dynamic_loss' in deformed_gaussians:
                        static_dynamic_weight = self.train_cfg.get('static_dynamic_weight', 0.5)
                        losses['loss_static_dynamic'] = deformed_gaussians['static_dynamic_loss'] * static_dynamic_weight

                    ## add for rigid-masking
                    if 'rigid_non_rigid_loss' in deformed_gaussians:
                        rigid_non_rigid_weight = self.train_cfg.get('rigid_non_rigid_loss', 0.5)
                        losses['loss_rigid_non_rigid'] = deformed_gaussians['rigid_non_rigid_loss'] * rigid_non_rigid_weight
            else:
                # No temporal frames, add temporal dimension for consistency with original temporal module
                means = means.unsqueeze(1)  # [B, 1, N, 3]
                quats = quats.unsqueeze(1)  # [B, 1, N, 4]
                scale = scale.unsqueeze(1)  # [B, 1, N, 3]
                opacity = opacity.unsqueeze(1)  # [B, 1, N, 1]
                        
        elif self.temporal_module is not None:
            # Original temporal module approach
            offsets = self.temporal_module(feature, [i['selected_frames'][:-1] for i in img_metas]) #[B, 10000, T, 3]
            if self.movement_regularizer is not None:
                losses['loss_movement'] = self.movement_regularizer(offsets, sem_pred)
            
            # means [B, 10000, 3]
            # sem_pred [B, 10000, num_classes]
            # self.dynamic_classes [2, 3, 4, 5, 6, 7, 9, 10]
            means = move_gaussians_temporal_module(means, sem_pred, offsets, self.dynamic_classes) #[B, T+1, 10000, 3]
        else:
            # No temporal module, add temporal dimension for consistency
            means = means.unsqueeze(1)  # [B, 1, N, 3]
            quats = quats.unsqueeze(1)  # [B, 1, N, 4]
            scale = scale.unsqueeze(1)  # [B, 1, N, 3]
            opacity = opacity.unsqueeze(1)  # [B, 1, N, 1]

        # Gaussian Splatting
        rendered_outs = {}
        vggt_feature_distillation_active = self._is_vggt_feature_distillation_active()
        if (self.training and self.use_vggt_feature_distillation and
                not vggt_feature_distillation_active and
                not self._vggt_distillation_wait_logged):
            print(
                "[VGGT Feature Distillation] Waiting until epoch "
                f"{self.vggt_feature_distillation_start_epoch}; "
                f"current epoch is {self.current_epoch}."
            )
            self._vggt_distillation_wait_logged = True
        if self.render_semantic and self.rasterizer is not None: # True, True
            # sem_pred needs to match the temporal dimension
            if means.ndim == 4:  # Temporal case [B, T, N, 3]
                sem_pred_temporal = sem_pred.unsqueeze(1).repeat(1, means.shape[1], 1, 1)  # [B, T, N, C]
            else:  # Static case [B, N, 3]
                sem_pred_temporal = sem_pred
            rendered_outs_sem = self.rasterizer(means, quats, scale, opacity, sem_pred_temporal, gs_intrins, gs_extrins, mode='RGB+D')
            rendered_outs['semantic'] = rendered_outs_sem[..., :-1]
            if self.render_depth:
                rendered_outs['depth'] = rendered_outs_sem[..., -1]
        
        # Feature rendering for VGGT feature distillation
        if vggt_feature_distillation_active:
            # Initialize feature distillation on first forward pass
            self._ensure_vggt_feature_distillation_initialized()
            
            # Predict per-Gaussian features
            gaussian_features = self.feature_head(feature)  # [B, N_gaussians, feature_dim]
            
            # Match temporal dimension like semantic rendering
            if means.ndim == 4:  # Temporal case [B, T_with_curr, N, 3]
                gaussian_features_temporal = gaussian_features.unsqueeze(1).repeat(1, means.shape[1], 1, 1)  # [B, T_with_curr, N_gaussians, feature_dim]
            else:  # Static case [B, N, 3]
                gaussian_features_temporal = gaussian_features
            
            # Render features through Gaussian splatting
            # Output shape: [B, T_with_curr, N_cams, H, W, feature_dim] if temporal, else [B, N_cams, H, W, feature_dim]
            rendered_features = self.rasterizer(
                means, quats, scale, opacity, 
                gaussian_features_temporal, 
                gs_intrins, gs_extrins, 
                mode='RGB'  # Use RGB mode for multi-channel features
            )
            rendered_outs['features'] = rendered_features
            
        
        if self.render_rgb and self.rasterizer is not None: # False, True
            rgb_pred = self.rgb_head(feature)
            # rgb_pred also needs to match temporal dimension
            if means.ndim == 4:  # Temporal case
                rgb_pred_temporal = rgb_pred.unsqueeze(1).repeat(1, means.shape[1], 1, 1)  # [B, T, N, C]
            else:  # Static case
                rgb_pred_temporal = rgb_pred
            if self.sh_degree > 0:
                B, T_or_N, N_or_C = rgb_pred_temporal.shape[:3]
                if means.ndim == 4:  # Temporal case
                    C = rgb_pred_temporal.shape[-1]
                    rendered_outs['rgb'] = self.rasterizer(means, quats, scale, opacity, rgb_pred_temporal.view(B, T_or_N, N_or_C, C//3, 3), gs_intrins,
                                                        gs_extrins, mode='RGB', sh_degree=self.sh_degree)
                else:  # Static case
                    C = rgb_pred_temporal.shape[-1]
                    rendered_outs['rgb'] = self.rasterizer(means, quats, scale, opacity, rgb_pred_temporal.view(B, T_or_N, C//3, 3), gs_intrins,
                                                        gs_extrins, mode='RGB', sh_degree=self.sh_degree)
            else:
                rendered_outs['rgb'] = self.rasterizer(means, quats, scale, opacity, rgb_pred_temporal, gs_intrins, gs_extrins, mode='RGB')

        # Compute losses
        if gs_gts is not None:
            losses_gs = self.rasterizer.calculate_losses(rendered_outs, gs_gts)
            
            # Apply temporal_weight to the main rendering losses
            if self.train_cfg is not None:
                temporal_weight = self.train_cfg.get('temporal_weight', 1.0)
                if temporal_weight != 1.0:
                    # Apply temporal weight to main losses (semantic and depth)
                    for loss_key in losses_gs:
                        if 'gs_sem' in loss_key or 'gs_depth' in loss_key:
                            losses_gs[loss_key] = losses_gs[loss_key] * temporal_weight
            
            losses.update(losses_gs)
        elif gs_gts_pixel is not None:
            t, ncams = gs_extrins.shape[1:3]
            rendered_outs = {k: v.unflatten(1, (t, ncams)) for k, v in rendered_outs.items()}
            losses_gs = self.rasterizer.calculate_losses_pixel(rendered_outs, gs_gts_pixel)
            
            # Apply temporal_weight to pixel losses as well
            if self.train_cfg is not None:
                temporal_weight = self.train_cfg.get('temporal_weight', 1.0)
                if temporal_weight != 1.0:
                    for loss_key in losses_gs:
                        if 'gs_sem' in loss_key or 'gs_depth' in loss_key:
                            losses_gs[loss_key] = losses_gs[loss_key] * temporal_weight
            
            losses.update(losses_gs)

        if (self.training and self.use_vggt_feature_distillation and
                not vggt_feature_distillation_active):
            warmup_loss = self._get_vggt_feature_distillation_warmup_loss()
            if warmup_loss is not None:
                losses['loss_vggt_feature_distillation_warmup'] = warmup_loss

        # VGGT feature distillation loss (Feature3DGS-style, only on current frame)
        if vggt_feature_distillation_active and 'features' in rendered_outs:
            # Extract VGGT teacher features for current frame

            # img_inputs_cur is list with 9, B, N, 3, H, W
            # img: [B, T, N, C, H, W], sensor2ego: [B, T, N, 4, 4], ego2global: [B, T, N, 4, 4], ego_l2global: [B, T, 4, 4],
            # cam2ego_l: [B, T, N, 4, 4], intrins: [B, T, N, 3, 3], # post_rot: [B, T, N, 3, 3], post_trans: [B, T, N, 3], bda: [B, 3, 3]
            imgs_cur = img_inputs_cur[0]
            if imgs_cur.ndim == 5:
                B, num_cams, C_img, H, W = imgs_cur.shape
            elif imgs_cur.ndim == 4:
                B = 1
                num_cams, C_img, H, W = imgs_cur.shape
            else:
                raise ValueError(f'Unexpected image tensor shape for VGGT distillation: {imgs_cur.shape}')
            imgs_flat = imgs_cur.reshape(B * num_cams, C_img, H, W)
            
            # Load or compute VGGT features with caching
            # Note: vggt_model will be loaded on-demand on first cache miss
            # # CRITICAL: Order of features matches order of imgs_flat
            # # imgs_flat order: [batch0_cam0, batch0_cam1, ..., batch0_cam5, 
            # #                   batch1_cam0, batch1_cam1, ..., batch1_cam5, ...]
            # # sample_idx mapping for verification:
            # if hasattr(self, '_debug_sample_matching') and self._debug_sample_matching:
            #     print(f"[VGGT Feature Matching Debug]")
            #     print(f"  Batch size: {B}, Cameras: {num_cams}")
            #     for b in range(min(B, 2)):  # Print first 2 batches only
            #         sample_idx = img_metas[b].get('sample_idx', 'N/A')
            #         print(f"  Batch {b}: sample_idx={sample_idx}")
            #         for cam_idx in range(num_cams):
            #             flat_idx = b * num_cams + cam_idx
            #             print(f"    imgs_flat[{flat_idx}] → Batch {b}, Camera {cam_idx}")
            
            with torch.no_grad():
                vggt_teacher_features = self.extract_vggt_teacher_features(
                    imgs_flat, 
                    num_cams=num_cams, 
                    img_metas=img_metas
                )  # List of [B*N, C_vggt, H_vggt, W_vggt] for each layer
            
            # Project VGGT features to compact space
            # vggt_teacher_features is a list of tensors, one per layer
            # Each element: [B*N, C_vggt, H_vggt, W_vggt]
            # Order PRESERVED from imgs_flat: [batch0_cam0, ..., batch0_cam5, batch1_cam0, ...]
            vggt_features_projected = []
            for layer_features in vggt_teacher_features:
                # layer_features: [B*N, C_vggt, H_vggt, W_vggt]
                B_N, C_vggt, H_vggt, W_vggt = layer_features.shape
                
                # Permute to [B*N, H_vggt, W_vggt, C_vggt]
                layer_features = layer_features.permute(0, 2, 3, 1)
                
                # Reshape to [B, N, H_vggt, W_vggt, C_vggt]
                layer_features_reshaped = layer_features.view(B, num_cams, H_vggt, W_vggt, C_vggt)
                
                # Project: [B, N, H_vggt, W_vggt, C_vggt] -> [B, N, H_vggt, W_vggt, gaussian_feature_dim]
                projected = self.feature_projector(layer_features_reshaped)
                vggt_features_projected.append(projected)
            
            # Use the last layer for distillation
            teacher_features = vggt_features_projected[-1]  # [B, N, H_vggt, W_vggt, feature_dim]
            
            # Extract current frame from rendered features
            # rendered_features_full shape: [B, T*N, H, W, C] where T*N = num_temporal * num_cams
            rendered_features_full = rendered_outs['features']
            
            
            # Reshape rendered features from [B, T*N, H, W, C] to [B, T, N, H, W, C]
            B_render, TN, H_render, W_render, C_render = rendered_features_full.shape
            T_with_curr = means.shape[1]  # Number of temporal frames (including current)
            
            # Reshape: [B, T*N, H, W, C] -> [B, T, N, H, W, C]
            rendered_features_reshaped = rendered_features_full.view(B, T_with_curr, num_cams, H_render, W_render, C_render)
            
            # Extract only current frame (last temporal index)
            # Current frame is at index -1 (last temporal frame)
            rendered_features_current = rendered_features_reshaped[:, -1:, :, :, :, :]  # [B, 1, N, H, W, feature_dim]
            
            
            # Compute feature distillation loss
            # Loss will handle spatial upsampling internally (teacher H' × W' -> rendered H × W)
            # Note: Loss expects teacher_features as [B, N, H, W, C] and will expand temporal dim internally
            feature_loss = self.feature_distillation_loss(
                rendered_features=rendered_features_current,  # [B, 1, N, H_render, W_render, feature_dim]
                teacher_features=teacher_features,  # [B, N, H_vggt, W_vggt, feature_dim]
            )
            losses['loss_vggt_feature_distillation'] = feature_loss
            
            # Optional: Temporal contrastive loss across frames
            if self.temporal_feature_loss is not None and means.ndim == 4 and rendered_features_full.shape[1] > 1:
                temporal_loss = self.temporal_feature_loss(rendered_features_full)
                losses['loss_temporal_feature'] = temporal_loss

        # Compute 3D losses
        if voxel_semantics is not None:
            # Voxelize Gaussians
            all_semantics = []
            all_densities = []

            if self.temporal_loss_3d:
                assert means.ndim == 4, "Means should be 4D tensor"
                for b in range(means.shape[0]):
                    means_b = means[b]
                    quats_b = quats[b]
                    scale_b = scale[b]
                    opacity_b = opacity[b]
                    sem_pred_b = sem_pred[b]
                    for t in range(means_b.shape[0]):
                        density, semantics = self.gaussians_to_occupancy(means_b[t], quats_b, scale_b, opacity_b, sem_pred_b)
                        all_densities.append(density)
                        all_semantics.append(semantics)
            else:
                if means.ndim > 3:
                    means = means[:, -1]
                for b in range(means.shape[0]):
                    density, semantics = self.gaussians_to_occupancy(means[b], quats[b], scale[b], opacity[b], sem_pred[b])
                    all_densities.append(density)
                    all_semantics.append(semantics)
            density = torch.stack(all_densities)
            semantics = torch.stack(all_semantics)

            voxel_semantics=voxel_semantics.long().reshape(-1)
            density = density.reshape(-1)
            semantics = semantics.reshape(-1, semantics.shape[-1])
            semantic_mask = voxel_semantics!=17
            if not self.with_others:
                semantic_mask = semantic_mask & (voxel_semantics != 0)
                voxel_semantics = voxel_semantics - 1
            if self.use_mask:
                mask_camera = mask_camera.reshape(-1).to(torch.bool)            
                combined_mask = (semantic_mask * mask_camera)
                loss_density = self.loss_occ_density(density[mask_camera], semantic_mask.float()[mask_camera])
                loss_semantic = self.loss_occ_semantics(semantics[combined_mask], voxel_semantics[combined_mask])
            else:
                loss_density = self.loss_occ_density(density, semantic_mask.float())
                loss_semantic = self.loss_occ_semantics(semantics[semantic_mask], voxel_semantics[semantic_mask])

            losses['loss_density_3d'] = loss_density
            losses['loss_semantic_3d'] = loss_semantic

        return losses
        
