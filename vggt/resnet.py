import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from vggt.models.vggt import VGGT
import torch.nn.functional as F
import torch.nn as nn

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
 

class FeatureDistiller(nn.Module):
    def __init__(self, teacher_channels, student_channels_list):

        super(FeatureDistiller, self).__init__()
        self.num_layers = len(student_channels_list)
        # Create a mapping block for each student feature where the teacher features are processed.
        self.teacher_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=1),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(teacher_channels, stu_ch, kernel_size=1),
                nn.BatchNorm2d(stu_ch)
            )
            for stu_ch in student_channels_list
        ])

    def forward(self, teacher_features_list, student_features_list):

        total_loss = 0.0
        for i in range(self.num_layers):
            teacher_feat = teacher_features_list[i]  # [N, teacher_channels, H_t, W_t]
            student_feat = student_features_list[i]  # [N, C_s, H_s, W_s]
            # Get the target spatial size from student feature
            _, _, H_s, W_s = student_feat.shape
            # Resize teacher feature to match student's spatial dimensions.
            teacher_resized = F.interpolate(teacher_feat, size=(H_s, W_s), mode='bilinear', align_corners=False)
            # Project teacher feature to student's channel dimensions via a deeper mapping block.
            teacher_projected = self.teacher_projs[i](teacher_resized)

            teacher_norm = F.normalize(teacher_projected, p=2, dim=1)
            student_norm = F.normalize(student_feat, p=2, dim=1)
            # Compute MSE loss between normalized projected teacher feature and normalized student feature.
            layer_loss = F.l1_loss(teacher_norm, student_norm)
            total_loss += layer_loss

        return total_loss

def extract_vggt_features(aggregated_tokens_list, ps_idx, layer_indices, B, S, patch_h, patch_w):

    features_list = []
    for layer in layer_indices:
        tokens = aggregated_tokens_list[layer][:, :, ps_idx:]
        vggt_feat = tokens.view(B * S, -1, tokens.size(-1))
        vggt_feat = vggt_feat.permute(0, 2, 1)
        vggt_feat = vggt_feat.reshape(vggt_feat.size(0), vggt_feat.size(1), patch_h, patch_w)
        features_list.append(vggt_feat)
    return features_list


@DETECTORS.register_module()
class CGFormerVGGT_Pure(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        # self.img_backbone = builder.build_backbone(img_backbone)

        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.depth_loss = depth_loss

        # self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.patch_size = 14
        vggt_model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        # self.distill = FeatureDistiller(2048, [48, 80, 224, 640, 2560])


        self.img_backbone = vggt_model.aggregator
        freeze_model(self.img_backbone)
        self.proj1 = nn.Sequential(nn.Conv2d(2048, 640, kernel_size=1),
                                    nn.BatchNorm2d(640),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(640, 640, kernel_size=1))
        
        self.proj2 = nn.Sequential(nn.Conv2d(2048, 640, kernel_size=1),
                                    nn.BatchNorm2d(640),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(640, 640, kernel_size=1))
        self.proj3 = nn.Sequential(nn.Conv2d(2048, 640, kernel_size=1),
                                    nn.BatchNorm2d(640),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(640, 640, kernel_size=1))
        self.proj4 = nn.Sequential(nn.Conv2d(2048, 640, kernel_size=1),
                                    nn.BatchNorm2d(640),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(640, 640, kernel_size=1))     

        self.proj5= nn.Sequential(nn.Conv2d(640, 640, kernel_size=1),
                                    nn.BatchNorm2d(640),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(640, 640, kernel_size=1))     

        self.train_cfg = train_cfg


    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x_img = x

        # torch.Size([1, 1, 3, 384, 1280])
        # Layer 1: torch.Size([1, 80, 48, 160])
        # Layer 2: torch.Size([1, 224, 24, 80])
        # Layer 3: torch.Size([1, 640, 12, 40])
        # Layer 4: torch.Size([1, 2560, 12, 40])

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x, x_img
    
    def extract_img_feat(self, img_inputs, img_metas):

        img = img_inputs[0]

        B, C, orig_height, orig_width = img.size()
        new_height = 14 * round(orig_height / 14)  # 14 * round(370 / 14) = 14 * 26 = 364
        new_width = 14 * round(orig_width / 14)     # 14 * round(1220 / 14) = 14 * 87 = 1218

        vggt_img = F.interpolate(img, size=(new_height, new_width),
                    mode='bilinear', align_corners=False).unsqueeze(1)
        
        B, S, _, H, W = vggt_img.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        with torch.no_grad():
            aggregated_tokens_list, ps_idx = self.img_backbone(vggt_img)
        
        layer_indices = [6, 12, 18, 23]
        img_enc_feats = extract_vggt_features(aggregated_tokens_list, ps_idx, layer_indices, B, S, patch_h, patch_w)

        new_feat = []
        for i in range(len(img_enc_feats)):
            new_feat.append(nn.functional.interpolate(img_enc_feats[i], size=(48, 160), mode='bilinear', align_corners=False)
                            )



        img_enc_feats = self.proj1(new_feat[0])+self.proj2(new_feat[1])+self.proj3(new_feat[2])+self.proj4(new_feat[3])
        img_enc_feats = self.proj5(img_enc_feats)



        img_enc_feats, x_img = self.image_encoder(img_inputs[0]) # torch.Size([1, 1, 640, 48, 160])
        B,C, H,W =img.size()

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats.unsqueeze(1)] + img_inputs[1:7] + [mlp_input], img_metas)
        #1, 1, 128, 48, 160
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)
        # torch.Size([1, 1, 128, 128, 16])

        if B > 1:
            x_list = []
            for i in range(B):
                camera_paras = img_inputs[1:7]
                camera_paras_batch = []
                
                for j in range(6):
                    camera_paras_batch.append(camera_paras[j][i:i+1])

                x = self.VoxFormer_head(
                    [context[i:i+1]],
                    proposal[i:i+1],
                    cam_params=camera_paras_batch,
                    lss_volume=coarse_queries[i:i+1],
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth[i:i+1].unsqueeze(1)]
                )
                x_list.append(x)
            x = torch.cat(x_list, dim=0)
        else:

            x = self.VoxFormer_head(
                [context],
                proposal,
                cam_params=img_inputs[1:7],
                lss_volume=coarse_queries,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth.unsqueeze(1)]
            )

        return x, depth
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        losses = dict()

        img_inputs = data_dict['img_inputs']
        img_inputs[0] = data_dict['img_aux']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']



        with torch.no_grad():
            img = data_dict['img_aux']

            B,C, orig_height, orig_width = img.size()
            new_height = 14 * round(orig_height / 14)  # 14 * round(370 / 14) = 14 * 26 = 364
            new_width = 14 * round(orig_width / 14)     # 14 * round(1220 / 14) = 14 * 87 = 1218

            vggt_img = F.interpolate(img, size=(new_height, new_width),
                        mode='bilinear', align_corners=False).unsqueeze(1)
            
            B, S, _, H, W = vggt_img.shape
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            # aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(vggt_img)
            aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(vggt_img)

            # Extract features from the last 4 layers
            layer_indices = [2, 6, 12, 18, 23]
    
            vggt_feat = extract_vggt_features(aggregated_tokens_list, ps_idx, layer_indices, B, S, patch_h, patch_w)

            if self.train_cfg['vggt_depth_distill']:
                depth_map, depth_conf = self.vggt_model.depth_head(aggregated_tokens_list, vggt_img, ps_idx)

            # save depthmap visualization
            # depth_map = depth_map.squeeze()[0].cpu().detach().numpy()
            # plt.imshow(depth_map.squeeze()[0].cpu().detach().numpy()) #save
            # plt.show()
            # plt.savefig('depth_map2.png')




        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)

        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        # if len(voxel_feats_enc) > 1:
            # voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

    
        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)
        

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_inputs[0] = data_dict['img_aux']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)