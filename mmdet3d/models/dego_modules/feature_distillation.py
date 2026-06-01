import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """Project high-dimensional teacher features (DINOv2) to low-dimensional Gaussian features.
    
    This creates a compact feature representation that can be efficiently rendered through
    Gaussians. Following Feature3DGS, we use a modest feature dimension (16-32) with a
    simple 1x1 convolution projection.
    
    Args:
        teacher_dim (int): Dimension of teacher features (e.g., 768 for DINOv2-base). Default: 768
        gaussian_feature_dim (int): Dimension of Gaussian feature vectors. Default: 32
        use_batch_norm (bool): Whether to use batch normalization. Default: False
    """
    
    def __init__(self, 
                 teacher_dim=768,
                 gaussian_feature_dim=32,
                 use_batch_norm=False):
        super().__init__()
        
        self.teacher_dim = teacher_dim
        self.gaussian_feature_dim = gaussian_feature_dim
        
        # Simple 1x1 projection (can be interpreted as conv or linear depending on input shape)
        layers = []
        layers.append(nn.Linear(teacher_dim, gaussian_feature_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(gaussian_feature_dim))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, teacher_features):
        """Project teacher features to compact Gaussian feature space.
        
        Args:
            teacher_features: [B, num_cams, H, W, C_teacher] - Teacher features (e.g., DINOv2)
            
        Returns:
            projected_features: [B, num_cams, H, W, C_gaussian] - Projected features
        """
        B, num_cams, H, W, C = teacher_features.shape
        assert C == self.teacher_dim, f"Expected teacher_dim={self.teacher_dim}, got {C}"
        
        # Reshape for projection
        features_flat = teacher_features.reshape(B * num_cams * H * W, C)
        projected = self.projector(features_flat)
        projected = projected.reshape(B, num_cams, H, W, -1)
        
        return projected
    
    def __repr__(self):
        return (f"FeatureProjector(teacher_dim={self.teacher_dim}, "
                f"gaussian_feature_dim={self.gaussian_feature_dim})")


class FeatureDistillationLoss(nn.Module):
    """Compute feature distillation loss between rendered Gaussian features and teacher features.
    
    This module implements the core idea from Feature3DGS: render compact feature vectors
    through Gaussians and align them with teacher features (DINOv2) at visible pixels.
    This provides dense, stable supervision beyond noisy class labels.
    
    Benefits:
    - Dense supervision: Every visible pixel gets feature supervision
    - Stable gradients: Teacher features are more robust than noisy pseudo-labels
    - Better thin structures: Feature matching helps with fine-grained geometry
    - Long-range consistency: Foundation features capture semantic relationships
    
    Args:
        loss_weight (float): Weight for this loss term. Default: 1.0
        loss_type (str): Type of loss - 'l2', 'huber', 'cosine'. Default: 'huber'
        huber_delta (float): Delta parameter for Huber loss. Default: 1.0
        normalize_features (bool): Whether to L2-normalize features before comparison. Default: True
        use_valid_mask (bool): Whether to mask out invalid pixels (outside image bounds). Default: True
    """
    
    def __init__(self, 
                 loss_weight=1.0,
                 loss_type='huber',
                 huber_delta=1.0,
                 normalize_features=True,
                 use_valid_mask=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.normalize_features = normalize_features
        self.use_valid_mask = use_valid_mask
    
    def forward(self, rendered_features, teacher_features, valid_mask=None):
        """Compute feature distillation loss.
        
        Args:
            rendered_features: [B, T_with_curr, num_cams, H_render, W_render, C] - Rendered Gaussian features
            teacher_features: [B, num_cams, H_teacher, W_teacher, C] - Teacher features (projected to same dim)
            valid_mask: [B, T, num_cams, H, W] - Optional mask for valid pixels
            
        Returns:
            loss: Scalar tensor - feature distillation loss
        """
        B, T_with_curr, num_cams, H_render, W_render, C = rendered_features.shape
        H_teacher, W_teacher = teacher_features.shape[2:4]
        
        # Spatially upsample teacher features to match rendered feature resolution if needed
        if H_teacher != H_render or W_teacher != W_render:
            # Reshape for interpolation: [B, num_cams, H, W, C] -> [B*num_cams, C, H, W]
            teacher_reshaped = teacher_features.permute(0, 1, 4, 2, 3).reshape(B * num_cams, C, H_teacher, W_teacher)
            # Bilinear interpolation to match rendered resolution
            teacher_upsampled = F.interpolate(teacher_reshaped, size=(H_render, W_render), 
                                             mode='bilinear', align_corners=False)
            # Reshape back: [B*num_cams, C, H, W] -> [B, num_cams, H, W, C]
            teacher_features = teacher_upsampled.reshape(B, num_cams, C, H_render, W_render).permute(0, 1, 3, 4, 2)
        
        # Expand teacher features to match temporal dimension
        teacher_features_expanded = teacher_features.unsqueeze(1).expand(B, T_with_curr, num_cams, H_render, W_render, C)

        # Normalize features if requested (cosine similarity-like)
        if self.normalize_features:
            rendered_features = F.normalize(rendered_features, dim=-1)
            teacher_features_expanded = F.normalize(teacher_features_expanded, dim=-1)
        
        # Compute loss based on type
        if self.loss_type == 'l2':
            loss = F.mse_loss(rendered_features, teacher_features_expanded, reduction='none')
            loss = loss.mean(dim=-1)  # Average over feature dimension
        elif self.loss_type == 'huber':
            loss = F.huber_loss(rendered_features, teacher_features_expanded, 
                               delta=self.huber_delta, reduction='none')
            loss = loss.mean(dim=-1)  # Average over feature dimension
        elif self.loss_type == 'cosine':
            # Cosine similarity loss: 1 - cosine_similarity
            cos_sim = (rendered_features * teacher_features_expanded).sum(dim=-1)
            loss = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Apply valid mask if provided
        if valid_mask is not None and self.use_valid_mask:
            loss = loss * valid_mask
            num_valid = valid_mask.sum() + 1e-6
            loss = loss.sum() / num_valid
        else:
            loss = loss.mean()
        return loss * self.loss_weight
    
    def __repr__(self):
        return (f"FeatureDistillationLoss(loss_weight={self.loss_weight}, "
                f"loss_type={self.loss_type}, normalize={self.normalize_features})")


class TemporalFeatureContrastiveLoss(nn.Module):
    """Compute temporal contrastive loss on rendered features across adjacent frames.
    
    This encourages feature consistency across time for static objects while allowing
    dynamics to vary. It's inspired by contrastive learning and helps with temporal
    coherence of the learned features.
    
    The idea: Features at the same 3D location (transformed across frames) should be
    similar (positives), while features at different locations should be different (negatives).
    
    Args:
        loss_weight (float): Weight for this loss term. Default: 0.1
        temperature (float): Temperature for contrastive loss. Default: 0.07
        num_negatives (int): Number of negative samples per positive. Default: 128
        use_hard_negatives (bool): Whether to use hard negative mining. Default: True
    """
    
    def __init__(self, 
                 loss_weight=0.1,
                 temperature=0.07,
                 num_negatives=128,
                 use_hard_negatives=True):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.use_hard_negatives = use_hard_negatives
    
    def forward(self, rendered_features, static_mask=None):
        """Compute temporal contrastive loss.
        
        Args:
            rendered_features: [B, T, num_cams, H, W, C] - Rendered features across time
            static_mask: [B, num_cams, H, W] - Optional mask indicating static regions
            
        Returns:
            loss: Scalar tensor - temporal contrastive loss
        """
        B, T, num_cams, H, W, C = rendered_features.shape
        
        if T < 2:
            # Need at least 2 frames for temporal contrast
            return torch.tensor(0.0, device=rendered_features.device)
        
        # Use current frame (t=T//2 or last frame) as anchor
        anchor_frame_idx = T - 1  # Use the current frame as anchor
        anchor_features = rendered_features[:, anchor_frame_idx]  # [B, num_cams, H, W, C]
        
        # Use adjacent frame as positive
        positive_frame_idx = max(0, anchor_frame_idx - 1)
        positive_features = rendered_features[:, positive_frame_idx]  # [B, num_cams, H, W, C]
        
        # Flatten spatial dimensions
        anchor_flat = anchor_features.reshape(B, num_cams * H * W, C)
        positive_flat = positive_features.reshape(B, num_cams * H * W, C)
        
        # Normalize for cosine similarity
        anchor_flat = F.normalize(anchor_flat, dim=-1)
        positive_flat = F.normalize(positive_flat, dim=-1)
        
        # Compute positive similarities
        pos_sim = (anchor_flat * positive_flat).sum(dim=-1) / self.temperature  # [B, N]
        
        # Sample negatives from different spatial locations
        N = anchor_flat.shape[1]
        num_neg = min(self.num_negatives, N)
        
        losses = []
        for b in range(B):
            # Random negative samples
            neg_indices = torch.randperm(N, device=anchor_flat.device)[:num_neg]
            negative_features = anchor_flat[b, neg_indices]  # [num_neg, C]
            
            # Compute negative similarities for all anchors
            neg_sim = torch.matmul(anchor_flat[b], negative_features.T) / self.temperature  # [N, num_neg]
            
            # InfoNCE loss
            logits = torch.cat([pos_sim[b:b+1].T, neg_sim], dim=1)  # [N, 1 + num_neg]
            labels = torch.zeros(N, dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
        
        loss = torch.stack(losses).mean()
        
        return loss * self.loss_weight
    
    def __repr__(self):
        return (f"TemporalFeatureContrastiveLoss(loss_weight={self.loss_weight}, "
                f"temperature={self.temperature}, num_negatives={self.num_negatives})")
