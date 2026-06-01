# DeGO 30-epoch release config. Cached VGGT distillation starts at epoch 21.

_base_ = ['./_base_/default_runtime.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

pc_range = [-40., -40., -1.0, 40., 40., 5.4]
eval_threshold_range = [.05]

dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
gt_root = 'data/gts'
mask_gt_root = 'data/grounded_sam_nusc'
depth_gt_root = 'data/metric_3d_nusc'
file_client_args = dict(backend='disk')

batch_size = 4

# Pseudo-label rendering resolution.
raster_downscale_factor = .44
raster_crop_top = 140
raster_shape = (int(data_config['src_size'][0] * raster_downscale_factor - raster_crop_top),
                int(data_config['src_size'][1] * raster_downscale_factor))

hidden_dim = 256
multi_adj_frame_id_cfg = (1, 8, 1)

T = 8
temporal_frame_ids = list(range(-T, T + 1, 1))
num_frames = 8

model = dict(
    type='DeGOVGGT',
    eval_threshold_range=eval_threshold_range,
    voxel_grid_cfg=grid_config,
    gaussian_init_scale=4,
    in_channels=hidden_dim,
    temporal_frame_ids=temporal_frame_ids,
    move_dynamic_gaussians=True,

    # Render Gaussian features and align them with cached VGGT teacher features.
    use_vggt_feature_distillation=True,
    vggt_feature_distillation_start_epoch=21,
    vggt_feature_distillation_config=dict(
        layer_indices=[22],
        gaussian_feature_dim=32,
        teacher_dim=2048,
        num_head_layers=2,
        projector=dict(use_batch_norm=False),
        distillation_loss=dict(
            loss_weight=1.0,
            loss_type='cosine',
            huber_delta=1.0,
            normalize_features=True,
            use_valid_mask=False,
        ),
        use_temporal_contrast=False,
        temporal_loss=dict(
            loss_weight=0.1,
            temperature=0.07,
            num_negatives=128,
            use_hard_negatives=True,
        ),
        use_cache=True,
        cache_max_size=1000,
        cache_dir='data/vggt_cache_spatial_temporal_block22',
        cache_selected_only=True,
        cache_use_fp16=False,
        cache_compress=False,
    ),

    train_cfg=dict(
        temporal_weight=1.0,
        deformation_weight=10.0,
        static_dynamic_weight=0.5,
    ),

    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=hidden_dim,
        start_level=0,
        num_outs=3),

    gaussian_decoder=dict(
        type='TemporalGaussianDecoder',
        in_channels=hidden_dim,
        temporal_frame_ids=temporal_frame_ids,

        gaussian_decoder_cfg=dict(
            type='GaussianDecoder',
            in_channels=hidden_dim,
            n_blocks_=3,
            pos_enc_cfg=dict(type='PointPositionalEncoding', out_channels=hidden_dim),
            temporal_att_cfg=dict(type='InducedGaussianAttention', in_channels=hidden_dim),
            self_att_cfg=dict(type='InducedGaussianAttention', in_channels=hidden_dim),
            cross_att_cfg=dict(type='GaussianImageCrossAttention', in_channels=hidden_dim),
            rect_cfg=dict(type='MeanRectifier', in_channels=hidden_dim),
            operation_order=('temporal_att', 'self_att', 'cross_att', 'rect')
        ),

        deformation_cfg=dict(
            type='GaussianDeformationMaskStaticRigid',
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            depth=6,
            time_channels=32,
            pos_enc_levels=6,
            time_enc_levels=4,
            apply_rotation_composition=True,
            deform_position=True,
            deform_rotation=True,
            deform_scale=True,
            deform_opacity=True,
            use_static_mask=False,
            use_rigid_mask=True,
        )
    ),

    rasterizer=dict(
        type='DeGORasterizer',
        raster_shape=raster_shape,
        depth_lw=.05,
        sem_lw=2.,
    ),
)

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='DeGOGeneratePseudoLabelsHorizon',
        downscale_factor=raster_downscale_factor,
        crop_top=raster_crop_top,
        num_frames=num_frames,
        grounded_sam_root=mask_gt_root,
        depth_root=depth_gt_root,
        temporal_frame_ids=temporal_frame_ids),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gs_gts', 'gs_intrins', 'gs_extrins'])
]

val_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]

test_pipeline = val_pipeline

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    temporal_frame_ids=temporal_frame_ids,
    eval_threshold_range=eval_threshold_range,
)


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=12,
    train=dict(
        data_root=data_root,
        ann_file='data/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        gt_root=gt_root,
        box_type_3d='LiDAR',
    ),

    val=dict(
        pipeline=val_pipeline,
        ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
        gt_root=gt_root,
    ),
    test=dict(
        pipeline=test_pipeline,
        ann_file='data/bevdetv2-nuscenes_infos_val.pkl',
        gt_root=gt_root,
    ),
)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(
        custom_keys={
            'deformation_module': dict(lr_mult=1.0),
            'time_embed': dict(lr_mult=1.0),
            'feature_head': dict(lr_mult=1.0),
            'feature_projector': dict(lr_mult=1.0),
        }
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='CustomCosineAnealing',
    start_at=9,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    min_lr_ratio=1e-2
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='CustomTensorboardLoggerHook',
            log_dir=None,
            interval=10,
            min_grad_norm=0.01,
        ),
    ])

runner = dict(type='EpochBasedRunner', max_epochs=30)

# Evaluate every six epochs.
evaluation = dict(interval=6)

custom_hooks = [
    dict(type='DeGOSetEpochInfoHook', priority='VERY_HIGH'),
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=1
    ),
    dict(
        type='WeightMonitorHook',
        interval=50,
        track_weight_updates=True,
        monitor_layers=['img_backbone', 'img_neck', 'gaussian_decoder'],
        min_norm_threshold=1e-6,
        priority='LOW'
    ),
]

checkpoint_config = dict(interval=1)
