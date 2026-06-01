# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn


def freeze_model(model):
    """Freeze all parameters in a model to prevent training.
    
    Args:
        model (nn.Module): The model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def extract_vggt_features(aggregated_tokens_list, ps_idx, layer_indices, B, S, patch_h, patch_w):
    """Extract spatial features from VGGT aggregated tokens at specified layers.
    
    VGGT outputs tokens in format [B, S, N, C] where N includes special tokens
    and patch tokens. This function:
    1. Selects specific layers
    2. Removes special tokens (keeps only patch tokens)
    3. Reshapes to spatial feature maps
    
    Args:
        aggregated_tokens_list (list[Tensor]): Token tensors from each VGGT layer
        ps_idx (int): Patch start index (index where patch tokens begin)
        layer_indices (list[int]): Which layers to extract features from
        B (int): Batch size
        S (int): Sequence length (number of frames/views)
        patch_h (int): Height in patches
        patch_w (int): Width in patches
        
    Returns:
        list[Tensor]: Extracted features [B*S, C, H, W] for each selected layer
    """
    features_list = []
    for layer_idx in layer_indices:
        # Get tokens from this layer and remove special tokens
        tokens = aggregated_tokens_list[layer_idx][:, :, ps_idx:]  # [B, S, N_patches, C]
        
        # Reshape to spatial grid
        # Combine batch and sequence dimensions
        vggt_feat = tokens.view(B * S, -1, tokens.size(-1))  # [B*S, N_patches, C]
        vggt_feat = vggt_feat.permute(0, 2, 1)  # [B*S, C, N_patches]
        
        # Reshape patches to 2D spatial grid
        vggt_feat = vggt_feat.reshape(
            vggt_feat.size(0), 
            vggt_feat.size(1), 
            patch_h, 
            patch_w
        )  # [B*S, C, H, W]
        
        features_list.append(vggt_feat)
    
    return features_list
