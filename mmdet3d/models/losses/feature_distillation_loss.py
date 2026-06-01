import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class FeatureDistillationLoss(nn.Module):
    """Feature distillation loss between teacher and student features.
    
    This loss aligns student features with teacher features through projection
    and normalized comparison. Supports multiple feature levels.
    
    Args:
        teacher_channels (int): Number of channels in teacher features
        student_channels_list (list[int]): Channel numbers for each student feature level
        loss_weight (float): Overall weight for the distillation loss. Default: 1.0
        distill_type (str): Type of distance metric - 'l1', 'l2', or 'cosine'. Default: 'l1'
        normalize_features (bool): Whether to L2-normalize features. Default: True
    """
    
    def __init__(self, 
                 teacher_channels,
                 student_channels_list,
                 loss_weight=1.0,
                 distill_type='l1',
                 normalize_features=True):
        super(FeatureDistillationLoss, self).__init__()
        
        self.loss_weight = loss_weight
        self.distill_type = distill_type
        self.normalize_features = normalize_features
        self.num_layers = len(student_channels_list)
        
        # Create projection layers to map teacher features to student dimensions
        self.teacher_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
                nn.BatchNorm2d(teacher_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(teacher_channels // 2, stu_ch, kernel_size=1),
                nn.BatchNorm2d(stu_ch)
            )
            for stu_ch in student_channels_list
        ])
        
    def forward(self, teacher_features_list, student_features_list):
        """Compute distillation loss between teacher and student features.
        
        Args:
            teacher_features_list (list[Tensor]): Teacher features, each [N, C_t, H_t, W_t]
            student_features_list (list[Tensor]): Student features, each [N, C_s, H_s, W_s]
            
        Returns:
            Tensor: Weighted distillation loss (scalar)
        """
        assert len(teacher_features_list) == len(student_features_list) == self.num_layers, \
            f"Expected {self.num_layers} feature levels, got teacher={len(teacher_features_list)}, student={len(student_features_list)}"
        
        total_loss = 0.0
        
        for i in range(self.num_layers):
            teacher_feat = teacher_features_list[i]  # [N, C_t, H_t, W_t]
            student_feat = student_features_list[i]  # [N, C_s, H_s, W_s]
            
            # Resize teacher to match student spatial dimensions
            _, _, H_s, W_s = student_feat.shape
            teacher_resized = F.interpolate(
                teacher_feat, 
                size=(H_s, W_s), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Project teacher to student channels
            teacher_projected = self.teacher_projs[i](teacher_resized)
            
            # Normalize if needed
            if self.normalize_features:
                teacher_projected = F.normalize(teacher_projected, p=2, dim=1)
                student_feat = F.normalize(student_feat, p=2, dim=1)
            
            # Compute loss based on type
            if self.distill_type == 'l1':
                layer_loss = F.l1_loss(teacher_projected, student_feat)
            elif self.distill_type == 'l2':
                layer_loss = F.mse_loss(teacher_projected, student_feat)
            elif self.distill_type == 'cosine':
                # Cosine similarity loss: 1 - mean(cosine_similarity)
                cos_sim = F.cosine_similarity(
                    teacher_projected.flatten(2), 
                    student_feat.flatten(2), 
                    dim=1
                ).mean()
                layer_loss = 1 - cos_sim
            else:
                raise ValueError(f"Unknown distill_type: {self.distill_type}")
            
            total_loss += layer_loss
        
        # Average over layers and apply weight
        return total_loss * self.loss_weight / self.num_layers
