# Inspired by 4DGaussians deformation module for dynamic 3D Gaussian Splatting

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.runner.base_module import BaseModule
import numpy as np


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions. Imported from pytorch3d"""
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def positional_encoding(input_data, pos_enc_freqs):
    """Apply positional encoding to input data"""
    input_data_emb = (input_data.unsqueeze(-1) * pos_enc_freqs).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb


@FEEDFORWARD_NETWORK.register_module()
class GaussianDeformationMaskStaticRigid(BaseModule):
    """
    Deformation module for dynamic 3D Gaussian Splatting inspired by 4DGaussians.
    This module predicts how Gaussians deform over time.
    """
    
    def __init__(self, 
                 in_channels=128,
                 hidden_dim=256, 
                 depth=6,
                 time_channels=32,
                 pos_enc_levels=6,
                 time_enc_levels=4,
                 apply_rotation_composition=True,
                 deform_position=True,
                 deform_rotation=True,
                 deform_scale=True,
                 deform_opacity=True,
                 use_static_mask=True,
                 use_rigid_mask=True):
        super(GaussianDeformationMaskStaticRigid, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.time_channels = time_channels
        self.apply_rotation_composition = apply_rotation_composition
        self.deform_position = deform_position
        self.deform_rotation = deform_rotation 
        self.deform_scale = deform_scale
        self.deform_opacity = deform_opacity
        self.use_static_mask = use_static_mask
        self.use_rigid_mask = use_rigid_mask
        
        # Positional encoding frequencies
        self.register_buffer('pos_enc_freqs', torch.FloatTensor([(2**i) for i in range(pos_enc_levels)]))
        self.register_buffer('time_enc_freqs', torch.FloatTensor([(2**i) for i in range(time_enc_levels)]))
        
        # Time embedding network
        time_input_dim = 2 * time_enc_levels + 1  # for encoded time
        self.time_embed = nn.Sequential(
            nn.Linear(time_input_dim, time_channels),
            nn.ReLU(),
            nn.Linear(time_channels, time_channels)
        )
        
        # Feature processing network
        pos_input_dim = 3 + 3 * pos_enc_levels * 2  # for encoded positions
        feature_input_dim = in_channels + pos_input_dim + time_channels
        
        # Main deformation network
        layers = []
        layers.append(nn.Linear(feature_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for i in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.feature_net = nn.Sequential(*layers)
        
        # Static/dynamic mask predictor
        if self.use_static_mask:
            self.static_mask_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        if self.use_rigid_mask:
            self.rigid_mask_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        # Deformation heads
        if self.deform_position:
            self.position_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
        
        if self.deform_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)
            )
        
        if self.deform_scale:
            self.scale_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )
            
        if self.deform_opacity:
            self.opacity_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, gaussians_means, gaussians_features, time_stamps, 
                gaussians_rotations=None, gaussians_scales=None, gaussians_opacity=None):
        """
        Forward pass of the deformation module.
        
        Args:
            gaussians_means: [B, N, 3] - Canonical Gaussian positions
            gaussians_features: [B, N, C] - Gaussian features from decoder
            time_stamps: [B, T] - Time stamps for each frame 
            gaussians_rotations: [B, N, 4] - Canonical rotations (optional)
            gaussians_scales: [B, N, 3] - Canonical scales (optional) 
            gaussians_opacity: [B, N, 1] - Canonical opacity (optional)
            
        Returns:
            Deformed Gaussian properties for each timestamp
        """
        B, N, _ = gaussians_means.shape
        T = time_stamps.shape[1]
        device = gaussians_means.device

        # Prepare outputs for all timesteps
        deformed_means = gaussians_means.unsqueeze(2).repeat(1, 1, T, 1)  # [B, N, T, 3]
        if gaussians_rotations is not None:
            deformed_rotations = gaussians_rotations.unsqueeze(2).repeat(1, 1, T, 1)  # [B, N, T, 4]
        if gaussians_scales is not None:
            deformed_scales = gaussians_scales.unsqueeze(2).repeat(1, 1, T, 1)  # [B, N, T, 3]
        if gaussians_opacity is not None:
            deformed_opacity = gaussians_opacity.unsqueeze(2).repeat(1, 1, T, 1)  # [B, N, T, 1]
        
        # Process each timestamp
        for t_idx in range(T):
            # Get time for this timestamp  
            current_time = time_stamps[:, t_idx:t_idx+1]  # [B, 1]

            # Encode time
            time_encoded = positional_encoding(current_time, self.time_enc_freqs)  # [B, time_dim]
            time_features = self.time_embed(time_encoded)  # [B, time_channels]
            
            # Encode positions
            pos_encoded = positional_encoding(gaussians_means, self.pos_enc_freqs)  # [B, N, pos_dim]
            
            # Expand time features to match Gaussian dimension
            time_features_expanded = time_features.unsqueeze(1).repeat(1, N, 1)  # [B, N, time_channels]
            
            # Concatenate all features
            combined_features = torch.cat([
                gaussians_features,  # [B, N, C]
                pos_encoded,         # [B, N, pos_dim] 
                time_features_expanded  # [B, N, time_channels]
            ], dim=-1)
            
            # Pass through feature network
            hidden_features = self.feature_net(combined_features)  # [B, N, hidden_dim]
            
            # Compute static/dynamic mask
            if self.use_static_mask:
                dynamic_mask = self.static_mask_head(hidden_features)  # [B, N, 1]
            else:
                dynamic_mask = torch.ones(B, N, 1, device=device)
            
            if self.use_rigid_mask:
                non_rigid_mask = self.rigid_mask_head(hidden_features)  # [B, N, 1]
            else:
                non_rigid_mask = torch.ones(B, N, 1, device=device)
            # breakpoint()
            # ## check the percentage of non-rigid gaussians with value > 0.1

            # print( "Non-rigid mask > 0.1 percentage:",
            #     (non_rigid_mask > 0.1).float().mean().item()
            # )
            # breakpoint()
            # Compute deformations with numerical stability
            if self.deform_position:
                pos_delta = self.position_head(hidden_features)  # [B, N, 3]
                # Clamp position delta to prevent extreme deformations
                pos_delta = torch.clamp(pos_delta, -10.0, 10.0)
                deformed_means[:, :, t_idx, :] = gaussians_means + dynamic_mask * pos_delta
            
            if self.deform_rotation and gaussians_rotations is not None:
                rot_delta = self.rotation_head(hidden_features)  # [B, N, 4] 
                if self.apply_rotation_composition:
                    # Compose rotations via quaternion multiplication
                    rot_delta_normalized = F.normalize(rot_delta + 1e-8, dim=-1)  # Add eps for numerical stability
                    # Identity quaternion for static and rigid Gaussians
                    identity_quat = torch.tensor([1., 0., 0., 0.], device=device)
                    deformed_rotations[:, :, t_idx, :] = quaternion_raw_multiply(
                        gaussians_rotations, 
                        dynamic_mask * non_rigid_mask * rot_delta_normalized + 
                        (1 - dynamic_mask * non_rigid_mask) * identity_quat
                    )
                else:
                    # Additive rotation
                    deformed_rotations[:, :, t_idx, :] = F.normalize(
                        gaussians_rotations + dynamic_mask * non_rigid_mask * rot_delta + 1e-8, dim=-1
                    )
            
            if self.deform_scale and gaussians_scales is not None:
                scale_delta = self.scale_head(hidden_features)  # [B, N, 3]
                # Clamp scale delta to prevent extreme changes
                scale_delta = torch.clamp(scale_delta, -1.0, 1.0)
                deformed_scales[:, :, t_idx, :] = gaussians_scales + dynamic_mask * non_rigid_mask * scale_delta
            
            if self.deform_opacity and gaussians_opacity is not None:
                opacity_delta = self.opacity_head(hidden_features)  # [B, N, 1]
                deformed_opacity[:, :, t_idx, :] = torch.clamp(
                    gaussians_opacity + dynamic_mask * non_rigid_mask * opacity_delta, 0., 1.
                )
        
        # Compute regularization losses
        deformation_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        static_dynamic_loss = torch.tensor(0.0, device=device, requires_grad=False)
        rigid_non_rigid_loss = torch.tensor(0.0, device=device, requires_grad=False)

        if self.training:
            # Deformation regularization loss - encourages small, smooth deformations
            reg_loss_parts = []
            
            if self.deform_position:
                # L2 norm of position changes
                pos_deltas = deformed_means - gaussians_means.unsqueeze(2)  # [B, N, T, 3]
                reg_loss_parts.append(torch.mean(pos_deltas.pow(2)))
            
            if self.deform_rotation and gaussians_rotations is not None:
                # Rotation changes (quaternion distance from identity)
                rot_deltas = deformed_rotations - gaussians_rotations.unsqueeze(2)  # [B, N, T, 4]
                reg_loss_parts.append(torch.mean(rot_deltas.pow(2)))
            
            if self.deform_scale and gaussians_scales is not None:
                # Scale changes
                scale_deltas = deformed_scales - gaussians_scales.unsqueeze(2)  # [B, N, T, 3]
                reg_loss_parts.append(torch.mean(scale_deltas.pow(2)))
            
            if self.deform_opacity and gaussians_opacity is not None:
                # Opacity changes
                opacity_deltas = deformed_opacity - gaussians_opacity.unsqueeze(2)  # [B, N, T, 1]
                reg_loss_parts.append(torch.mean(opacity_deltas.pow(2)))
            
            if len(reg_loss_parts) > 0:
                deformation_reg_loss = sum(reg_loss_parts)
            else:
                deformation_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Static/dynamic mask loss (if enabled)
            if self.use_static_mask:
                # Encourage the mask to be close to 0 or 1 (binary)
                static_dynamic_loss = torch.mean(dynamic_mask * (1 - dynamic_mask))  # Binary entropy-like loss
            
            if self.use_rigid_mask:
                rigid_non_rigid_loss = torch.mean(non_rigid_mask * (1 - non_rigid_mask))
        
        # Prepare return dictionary
        result = {'means': deformed_means}
        
        if gaussians_rotations is not None:
            result['rotations'] = deformed_rotations
        if gaussians_scales is not None:
            result['scales'] = deformed_scales  
        if gaussians_opacity is not None:
            result['opacity'] = deformed_opacity
            
        # Add losses to the result
        result['deformation_reg_loss'] = deformation_reg_loss
        if self.use_static_mask:
            result['static_dynamic_loss'] = static_dynamic_loss
        if self.use_rigid_mask:
            result['rigid_non_rigid_loss'] = rigid_non_rigid_loss
        return result

