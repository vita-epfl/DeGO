#!/usr/bin/env python
"""
SPATIAL-TEMPORAL VGGT cache generation - INTERLEAVED spatial-temporal attention.

This version implements INTERLEAVED spatial-temporal processing, inspired by how
VGGT alternates between global and frame attention across layers.


Processing:
    1. Extract VGGT tokens for spatial input (all N cameras at t₀)
    2. Extract VGGT tokens for temporal input (all T frames for each camera)
    3. At each layer L:
       - If L is even: Use spatial tokens (cross-camera attention)
       - If L is odd: Use temporal tokens (cross-frame attention)
    4. Save interleaved features at reference frame t₀

Optimization strategies:
1. Store only selected layers (not all 24) - saves ~20-23x space
2. Use float16 instead of float32 - saves 50% space
3. Compress cache files - saves ~30-50% space
4. All of the above - saves ~50-70x total space!

Usage:
    # Store only selected layers (biggest savings!)
    python generate_vggt_cache.py --cache-selected-only

    # Use float16 precision
    python generate_vggt_cache.py --use-fp16

    # Use compression
    python generate_vggt_cache.py --compress

    # Combine all strategies (maximum savings!)
    python generate_vggt_cache.py --cache-selected-only --use-fp16 --compress
"""

import argparse
import os
import sys
import time
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from tqdm import tqdm

sys.path.insert(0, '.')

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.models.detectors.vggt_utils import extract_vggt_features


def parse_args():
    parser = argparse.ArgumentParser(description='Generate spatial-temporal VGGT feature cache')
    parser.add_argument('--config', 
                        default='configs/ours_vggt_spv_img_enc_lw1_cos_spatial_temporal.py',
                        help='training config file path')
    parser.add_argument('--cache-dir', 
                        default='data/vggt_cache_spatial_temporal',
                        help='directory to save cached features')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=1,
                        help='batch size for feature extraction')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        help='number of dataloader workers')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device to use')
    parser.add_argument('--skip-existing',
                        action='store_true',
                        help='skip samples that already have cached features')
    
    # Parallel processing support
    parser.add_argument('--start-idx',
                        type=int,
                        default=0,
                        help='start index for processing subset of dataset (for parallel processing)')
    parser.add_argument('--end-idx',
                        type=int,
                        default=-1,
                        help='end index for processing subset of dataset (for parallel processing, -1 means all)')
    
    # Optimization options
    parser.add_argument('--cache-selected-only',
                        action='store_true',
                        help='Cache only selected layers instead of all 24 (saves ~20-23x space!)')
    parser.add_argument('--use-fp16',
                        action='store_true',
                        help='Store features in float16 instead of float32 (saves 50%% space)')
    parser.add_argument('--compress',
                        action='store_true',
                        help='Use compression when saving (saves ~30-50%% space, slight slowdown)')
    
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override config settings')
    args = parser.parse_args()
    return args


def check_cache_exists(cache_dir, sample_idx, num_cams=6):
    """Check if all camera views for a sample are already cached."""
    for cam_idx in range(num_cams):
        cache_file = os.path.join(cache_dir, f'sample_{sample_idx}_cam_{cam_idx}.pt')
        if not os.path.exists(cache_file):
            return False
    return True


def save_vggt_features(cache_dir, sample_idx, cam_idx, features, img=None, use_fp16=False, compress=False):
    """Save VGGT features to disk with optimization options.
    
    Args:
        cache_dir (str): Directory to save cache files
        sample_idx (int): Sample index
        cam_idx (int): Camera index
        features (list[Tensor]): List of feature tensors [C, H, W] for each layer
        img (Tensor): Resized input image [C, H, W] used for feature extraction
        use_fp16 (bool): Convert to float16 to save space
        compress (bool): Use compression (slower but smaller files)
    """
    cache_file = os.path.join(cache_dir, f'sample_{sample_idx}_cam_{cam_idx}.pt')
    
    # Features should be on CPU with batch dimension: [1, C, H, W]
    features_cpu = [f.cpu().unsqueeze(0) for f in features]
    
    # Apply optimizations
    if use_fp16:
        features_cpu = [f.half() for f in features_cpu]  # float32 -> float16
    
    # Save both features and resized image (for verification)
    data_to_save = {
        'features': features_cpu,
        'img': img.cpu() if img is not None else None
    }
    
    try:
        if compress:
            # Use gzip compression level 6 (good balance of speed/size)
            # Note: This requires more CPU but saves significant disk space
            import gzip
            import io
            
            # Save to memory buffer first
            buffer = io.BytesIO()
            torch.save(data_to_save, buffer)
            buffer.seek(0)
            
            # Compress and write to disk
            with gzip.open(cache_file + '.gz', 'wb', compresslevel=6) as f:
                f.write(buffer.read())
        else:
            torch.save(data_to_save, cache_file)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to save {cache_file}: {e}")


def load_vggt_features_optimized(cache_file, use_fp16=False, compress=False):
    """Load VGGT features with optimization support.
    
    Returns features in original float32 format even if stored as float16.
    Returns tuple: (features, img) where img may be None for old format.
    """
    try:
        if compress and os.path.exists(cache_file + '.gz'):
            import gzip
            with gzip.open(cache_file + '.gz', 'rb') as f:
                data = torch.load(f, map_location='cpu')
        else:
            data = torch.load(cache_file, map_location='cpu')
        
        # Handle both old format (list) and new format (dict)
        if isinstance(data, dict):
            features = data['features']
            img = data.get('img', None)
        else:
            # Old format: just features
            features = data
            img = None
        
        # Convert back to float32 if needed
        if use_fp16:
            features = [f.float() for f in features]
        
        return features, img
        
    except Exception as e:
        print(f"[ERROR] Failed to load {cache_file}: {e}")
        return None, None


def extract_and_cache_batch(model, imgs, img_metas, cache_dir, layer_indices, 
                            num_cams=6, patch_size=14, device='cuda',
                            cache_selected_only=False, use_fp16=False, compress=False):
    """Extract VGGT features with TRUE INTERLEAVED SPATIAL-TEMPORAL processing.
    
    TRUE INTERLEAVED ARCHITECTURE (not just output selection):
    
    This implements proper alternating attention where spatial and temporal
    actually refine each other through a shared token representation.
    
    Algorithm:
        1. Extract patch tokens for all (t, cam) pairs
        2. Initialize token dict: tokens[(t, cam)] = [NumPatches, C]
        3. For each layer L in range(depth):
            if L % 2 == 0:  # SPATIAL BLOCK
                For each timestamp t:
                    Fuse tokens across all N cameras at time t
                    Update: tokens[(t, cam)] for all cam
            else:  # TEMPORAL BLOCK  
                For each camera cam:
                    Fuse tokens across all T timestamps
                    Align everything back to reference t₀
                    Update: tokens[(t, cam)] for all t
        4. Extract refined tokens at (t₀, each cam) and convert to features
    
    This creates TRUE deep fusion where:
    - Spatial layers see results of previous temporal fusion
    - Temporal layers see results of previous spatial fusion
    - Information flows bidirectionally across layers
    
    Args:
        model: VGGT model
        imgs (Tensor): Input images [B, T, N, C, H, W]
        img_metas (list[dict]): Metadata with sample_idx
        cache_dir (str): Directory to save cache
        layer_indices (list[int]): Which layers to extract
        num_cams (int): Number of cameras
        patch_size (int): VGGT patch size
        device (str): Device to use
        cache_selected_only (bool): Cache only selected layers
        use_fp16 (bool): Store as float16
        compress (bool): Use compression
        
    Returns:
        int: Number of samples cached
    """
    # Handle input shape: could be [B, T, N, C, H, W] or [B, N, C, H, W]
    if imgs.ndim == 6:
        # Temporal data: [B, T, N, C, H, W]
        B, T, N, C, H, W = imgs.shape
        has_temporal = True
    elif imgs.ndim == 5:
        # Single frame: [B, N, C, H, W]
        B, N, C, H, W = imgs.shape
        T = 1
        imgs = imgs.unsqueeze(1)  # Add temporal dim: [B, 1, N, C, H, W]
        has_temporal = False
    else:
        raise ValueError(f"Unexpected imgs shape: {imgs.shape}")
    
    # Resize to multiple of patch_size
    new_height = patch_size * round(H / patch_size)
    new_width = patch_size * round(W / patch_size)
    
    H_new, W_new = new_height, new_width
    patch_h, patch_w = H_new // patch_size, W_new // patch_size
    
    # =========================================================================
    # STEP 1: Extract patch tokens for all (t, cam) pairs
    # =========================================================================
    
    # Prepare all images and extract patch embeddings
    BTN = B * T * N
    imgs_flat = imgs.reshape(BTN, C, H, W).to(device)  # Move to device
    imgs_resized = F.interpolate(
        imgs_flat,
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Extract patch tokens using VGGT's patch_embed
    # Note: patch_embed handles normalization internally, so we pass raw images
    with torch.no_grad():
        patch_tokens = model.aggregator.patch_embed(imgs_resized)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
    
    # patch_tokens: [B*T*N, num_patches, C_token]
    num_patches, C_token = patch_tokens.shape[1], patch_tokens.shape[2]
    
    # Reshape to [B, T, N, num_patches, C_token]
    patch_tokens = patch_tokens.view(B, T, N, num_patches, C_token)
    
    # Get special tokens (camera and register tokens)
    # Note: In VGGT, these are [1, 2, num_tokens, embed_dim] where dim 1 is for first frame vs rest
    # For spatial-temporal processing, we use index 0 (first frame type)
    camera_token = model.aggregator.camera_token[0, 0]  # [1, C_token]
    register_token = model.aggregator.register_token[0, 0]  # [num_reg, C_token]
    patch_start_idx = model.aggregator.patch_start_idx
    
    # Get positional embeddings
    pos = None
    if model.aggregator.rope is not None:
        pos = model.aggregator.position_getter(1, patch_h, patch_w, device=device)
        # Add offset for special tokens
        if patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(1, patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)  # [1, num_tokens, 2]
    
    # Get attention blocks
    frame_blocks = model.aggregator.frame_blocks
    global_blocks = model.aggregator.global_blocks
    depth = len(frame_blocks)
    
    # Determine which layers to extract
    if cache_selected_only and len(layer_indices) > 0:
        extract_at_layers = layer_indices
    else:
        extract_at_layers = list(range(depth))
    
    # =========================================================================
    # STEP 2: TRUE INTERLEAVED PROCESSING
    # =========================================================================
    
    cached_count = 0
    ref_t = 0  # Reference timestamp
    
    for b in range(B):
        sample_idx = img_metas[b].get('sample_idx', None)
        if sample_idx is None:
            print(f"\n[WARNING] Sample {b} missing sample_idx, skipping")
            continue
        
        # Initialize token dictionary for this sample: tokens[(t, cam)]
        tokens_dict = {}
        for t in range(T):
            for cam in range(N):
                # Start with patch tokens for (t, cam)
                cam_tokens = patch_tokens[b, t, cam]  # [num_patches, C_token]
                
                # Prepend special tokens: [camera_token, register_tokens, patch_tokens]
                full_tokens = torch.cat([
                    camera_token,  # [1, C_token]
                    register_token,  # [num_reg, C_token]
                    cam_tokens  # [num_patches, C_token]
                ], dim=0)  # [num_total_tokens, C_token]
                
                tokens_dict[(t, cam)] = full_tokens
        
        # =========================================================================
        # STEP 3: Alternating spatial-temporal transformer loop
        # =========================================================================
        
        # CRITICAL: VGGT has 24 BLOCKS, each with frame + global attention (48 layers total)
        # We process in BLOCKS where each block = spatial + temporal attention
        # So layer_indices=[20,21,22,23] means blocks 20,21,22,23
        
        # Storage for BLOCK outputs (not layer outputs)
        # Each block processes both spatial AND temporal, then concatenates
        block_outputs = {}  # block_outputs[block_idx][(t, cam)] = concatenated tokens
        
        num_blocks = depth  # depth is number of blocks (24 for VGGT)
        
        for block_idx in range(num_blocks):
            # Each block does BOTH spatial and temporal (order doesn't matter for our case)
            
            # PART 1: SPATIAL attention (like VGGT's global attention)
            spatial_tokens_dict = {}
            for t in range(T):
                # Collect tokens from all cameras at time t
                tokens_at_t = [tokens_dict[(t, cam)] for cam in range(N)]
                tokens_tensor = torch.stack(tokens_at_t, dim=0)  # [N, num_tokens, C_token]
                
                # Apply global attention (across cameras)
                with torch.no_grad():
                    # Use VGGT's global block for this block index
                    spatial_output = global_blocks[block_idx](tokens_tensor, pos=pos)
                
                # Store spatial output
                for cam in range(N):
                    spatial_tokens_dict[(t, cam)] = spatial_output[cam]
            
            # PART 2: TEMPORAL attention (like VGGT's frame attention)
            temporal_tokens_dict = {}
            for cam in range(N):
                # Collect tokens from all timestamps for this camera
                # Use OUTPUT from spatial attention as input to temporal
                tokens_over_time = [spatial_tokens_dict[(t, cam)] for t in range(T)]
                tokens_tensor = torch.stack(tokens_over_time, dim=0)  # [T, num_tokens, C_token]
                
                # Apply frame attention (across time)
                with torch.no_grad():
                    # Use VGGT's frame block for this block index
                    temporal_output = frame_blocks[block_idx](tokens_tensor, pos=pos)
                
                # Store temporal output
                for t in range(T):
                    temporal_tokens_dict[(t, cam)] = temporal_output[t]
            
            # PART 3: Combine spatial + temporal outputs (like VGGT combines frame + global)
            # Store concatenated output for this block
            if block_idx in extract_at_layers:
                block_outputs[block_idx] = {}
                for t in range(T):
                    for cam in range(N):
                        # Get both spatial and temporal outputs for this (t, cam)
                        spatial_tok = spatial_tokens_dict[(t, cam)]  # [num_tokens, C_token]
                        temporal_tok = temporal_tokens_dict[(t, cam)]  # [num_tokens, C_token]
                        
                        # Concatenate along channel dimension (like VGGT)
                        # This will be used for feature extraction
                        block_outputs[block_idx][(t, cam)] = (spatial_tok, temporal_tok)
            
            # Update tokens_dict for next block (use temporal output as it's the final one)
            tokens_dict = temporal_tokens_dict
        
        # =========================================================================
        # STEP 4: Extract refined tokens at (ref_t, each cam) and save
        # =========================================================================
        
        imgs_resized_batch = imgs_resized.view(B, T, N, C, H_new, W_new)
        
        for cam_idx in range(N):
            cam_features = []
            
            for block_idx in extract_at_layers:
                # Get spatial and temporal outputs for this block
                # block_outputs[block_idx][(t, cam)] = (spatial_tokens, temporal_tokens)
                spatial_tokens, temporal_tokens = block_outputs[block_idx][(ref_t, cam_idx)]
                
                # Extract patch tokens (skip special tokens) from both
                spatial_patches = spatial_tokens[patch_start_idx:]  # [num_patches, C_token]
                temporal_patches = temporal_tokens[patch_start_idx:]  # [num_patches, C_token]
                
                # Concatenate along channel dimension: [num_patches, 2*C_token]
                # This matches VGGT's format where frame + global are concatenated
                combined_patches = torch.cat([temporal_patches, spatial_patches], dim=-1)
                
                # Reshape to spatial feature map: [num_patches, 2*C_token] → [2*C_token, H', W']
                C_combined = combined_patches.shape[-1]  # Should be 2048
                feature_map = combined_patches.permute(1, 0).view(C_combined, patch_h, patch_w)
                
                cam_features.append(feature_map)  # [2048, H', W']
            
            # Get reference image
            cam_img_resized = imgs_resized_batch[b, ref_t, cam_idx]  # [C, H', W']
            
            # Save interleaved features
            save_vggt_features(cache_dir, sample_idx, cam_idx, cam_features,
                             img=cam_img_resized, use_fp16=use_fp16, compress=compress)
            cached_count += 1
    
    return cached_count


def estimate_cache_size(num_samples=28130, num_cams=6, num_layers=24, 
                       cache_selected_only=False, selected_layers=1,
                       use_fp16=False, compress=False):
    """Estimate total cache size with different optimization strategies."""
    
    # Base size per feature tensor (approximate)
    # Shape: [1, 2048, 18, 50] for nuScenes typical resolution
    feature_size_mb = (1 * 2048 * 18 * 50 * 4) / (1024 * 1024)  # 4 bytes for float32
    
    layers = selected_layers if cache_selected_only else num_layers
    size_per_file = feature_size_mb * layers
    
    if use_fp16:
        size_per_file /= 2  # Half the size
    
    if compress:
        size_per_file *= 0.4  # ~60% compression ratio
    
    total_files = num_samples * num_cams
    total_size_gb = (size_per_file * total_files) / 1024
    
    return total_size_gb, size_per_file


def generate_cache(args):
    """Main function to generate cache for entire dataset."""
    
    # ==========================================
    # CRITICAL: Set all random seeds for determinism
    # ==========================================
    SEED = 42  # Fixed seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("=" * 80)
    print("SPATIAL-TEMPORAL VGGT Feature Cache Generation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Random seed: {SEED} (DETERMINISTIC MODE)")
    print("")
    print("Processing mode: SPATIAL-TEMPORAL (Two-stage refinement)")
    print("  Stage 1 (SPATIAL): Multi-view aggregation at each timestamp")
    print("    - Process all N cameras together per timestamp")
    print("    - VGGT aggregates spatial context across views")
    print("  Stage 2 (TEMPORAL): Temporal aggregation per camera")
    print("    - Process T timestamps together per camera")
    print("    - VGGT aggregates temporal motion context")
    print("  Output: Features with both spatial AND temporal context")
    print("")
    print("Optimization strategies:")
    print(f"  Cache selected layers only: {args.cache_selected_only}")
    print(f"  Use float16: {args.use_fp16}")
    print(f"  Use compression: {args.compress}")
    print("=" * 80)
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Build model
    print("\n[1/5] Loading VGGT model...")
    full_model = build_model(cfg.model)
    
    if not hasattr(full_model, 'vggt_model'):
        print("[ERROR] Model does not have VGGT! Check your config.")
        return
    
    vggt_model = full_model.vggt_model
    if vggt_model is None and hasattr(full_model, "_load_vggt_model"):
        print("   VGGT teacher is lazy-loaded in this config; loading it now...")
        full_model._load_vggt_model()
        vggt_model = full_model.vggt_model

    if vggt_model is None:
        raise RuntimeError(
            "Model has vggt_model=None and no cache-generator-compatible loader. "
            "Check use_vggt_feature_distillation_config or pass a VGGT config."
        )

    vggt_model = vggt_model.to(args.device)
    vggt_model.eval()
    
    patch_size = full_model.patch_size if hasattr(full_model, 'patch_size') else 14
    layer_indices = full_model.vggt_layer_indices if hasattr(full_model, 'vggt_layer_indices') else [20]
    
    num_layers = vggt_model.aggregator.depth if hasattr(vggt_model.aggregator, 'depth') else 24
    print(f"   VGGT loaded: {num_layers} layers")
    print(f"   Selected layers for training: {layer_indices}")
    print(f"   Patch size: {patch_size}")
    
    # Estimate cache size
    print("\n[2/5] Estimating cache size...")
    total_size, size_per_file = estimate_cache_size(
        cache_selected_only=args.cache_selected_only,
        selected_layers=len(layer_indices),
        use_fp16=args.use_fp16,
        compress=args.compress
    )
    print(f"   Estimated size per file: {size_per_file:.2f} MB")
    print(f"   Estimated total cache size: {total_size:.1f} GB")
    print(f"   Note: Spatial-temporal processing may be slightly larger due to two-stage refinement")
    
    # Build dataset
    print("\n[3/5] Building dataset...")
    # CRITICAL: Use test_mode=True to disable augmentation!
    # Cache should store features at original resolution, not augmented
    cfg.data.train.test_mode = True
    dataset = build_dataset(cfg.data.train)
    total_dataset_size = len(dataset)
    print(f"   Total dataset size: {total_dataset_size} samples")
    
    # Apply start/end indices for parallel processing
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_dataset_size
    
    # Validate indices
    start_idx = max(0, min(start_idx, total_dataset_size))
    end_idx = max(start_idx, min(end_idx, total_dataset_size))
    
    if args.start_idx is not None or args.end_idx is not None:
        print(f"   Processing subset: [{start_idx}, {end_idx}) = {end_idx - start_idx} samples")
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(start_idx, end_idx))
    else:
        print(f"   Processing full dataset: {total_dataset_size} samples")
    
    # CRITICAL for determinism
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
        seed=SEED
    )
    
    print(f"   Dataloader: {len(data_loader)} batches")
    
    # Process dataset
    print("\n[4/5] Extracting VGGT features (spatial-temporal)...")
    print("   This will take longer than single-stage processing")
    
    total_cached = 0
    total_skipped = 0
    total_time = 0
    num_cams = 6
    
    progress_bar = tqdm(data_loader, desc="Caching", unit="batch")
    
    for batch_idx, data_batch in enumerate(progress_bar):
        try:
            # Extract data
            img_metas = data_batch['img_metas'].data[0]
            img_inputs = data_batch['img_inputs']
            if isinstance(img_inputs, list):
                imgs = img_inputs[0].data if hasattr(img_inputs[0], 'data') else img_inputs[0]
            else:
                imgs = img_inputs.data if hasattr(img_inputs, 'data') else img_inputs
            
            # Handle temporal data shape
            # Expected: [B, T, N, C, H, W] for temporal sequences
            # Or: [B, N, C, H, W] if no temporal dimension
            if imgs.ndim == 6:
                B, T, N, C, H, W = imgs.shape
            elif imgs.ndim == 5:
                B, N, C, H, W = imgs.shape
                T = 1
            else:
                raise ValueError(f"Unexpected image shape: {imgs.shape}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to extract data from batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
        
        # Check if should skip
        if args.skip_existing:
            all_exist = True
            for meta in img_metas:
                sample_idx = meta.get('sample_idx', None)
                if sample_idx is None or not check_cache_exists(args.cache_dir, sample_idx, num_cams):
                    all_exist = False
                    break
            
            if all_exist:
                total_skipped += B
                progress_bar.set_postfix({
                    'cached': total_cached,
                    'skipped': total_skipped,
                    'avg_time': f'{total_time/(batch_idx+1):.2f}s'
                })
                continue
        
        # Extract and cache
        start_time = time.time()
        
        try:
            cached_count = extract_and_cache_batch(
                vggt_model,
                imgs,
                img_metas,
                args.cache_dir,
                layer_indices,
                num_cams=num_cams,
                patch_size=patch_size,
                device=args.device,
                cache_selected_only=args.cache_selected_only,
                use_fp16=args.use_fp16,
                compress=args.compress
            )
            total_cached += cached_count // num_cams
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        progress_bar.set_postfix({
            'cached': total_cached,
            'skipped': total_skipped,
            'batch_time': f'{batch_time:.2f}s',
            'avg_time': f'{total_time/(batch_idx+1):.2f}s'
        })
    
    progress_bar.close()
    
    # Summary
    print("\n[5/5] Cache generation complete!")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    cache_files = [f for f in os.listdir(args.cache_dir) if f.endswith('.pt') or f.endswith('.pt.gz')]
    total_samples = len(cache_files) // num_cams
    
    print(f"✓ Total samples cached: {total_cached}")
    print(f"✓ Total samples skipped: {total_skipped}")
    print(f"✓ Cache files created: {len(cache_files)}")
    print(f"✓ Samples in cache: {total_samples}")
    print(f"✓ Average time per batch: {total_time/len(data_loader):.2f}s")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"✓ Cache directory: {args.cache_dir}")
    
    # Actual disk usage
    if cache_files:
        import subprocess
        try:
            result = subprocess.run(['du', '-sh', args.cache_dir], capture_output=True, text=True)
            actual_size = result.stdout.split()[0]
            print(f"✓ Actual cache size: {actual_size}")
        except:
            pass
    
    print(f"\nOptimizations applied:")
    print(f"  ✓ Cache selected only: {args.cache_selected_only} (saves ~23x if enabled)")
    print(f"  ✓ Float16: {args.use_fp16} (saves 50% if enabled)")
    print(f"  ✓ Compression: {args.compress} (saves ~40% if enabled)")
    
    print("=" * 80)
    print("\n✓ Cache ready for training!")
    print(f"Update your config: cache_dir='{args.cache_dir}'")
    
    print("\n📝 SPATIAL-TEMPORAL FEATURES:")
    print("   - Features have multi-view spatial context (from all cameras)")
    print("   - Features have temporal motion context (across time)")
    print("   - Best for tasks requiring both spatial and temporal understanding")
    
    if args.cache_selected_only or args.use_fp16 or args.compress:
        print("\n⚠️  IMPORTANT: Update your model to use the same optimizations!")
        print("   See MEMORY_OPTIMIZATION_SUMMARY.md for details.")



if __name__ == '__main__':
    args = parse_args()
    generate_cache(args)
