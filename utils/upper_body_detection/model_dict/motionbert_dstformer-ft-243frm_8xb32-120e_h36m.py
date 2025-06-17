auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_root = 'data/h36m/'
dataset_type = 'Human36mDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        rule='less',
        save_best='MPJPE',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        att_fuse=True,
        depth=5,
        feat_size=512,
        in_channels=3,
        mlp_ratio=2,
        num_heads=8,
        seq_len=243,
        type='DSTFormer'),
    head=dict(
        decoder=dict(
            concat_vis=True,
            num_keypoints=17,
            rootrel=True,
            type='MotionBERTLabel'),
        embedding_size=512,
        in_channels=512,
        loss=dict(type='MPJPEVelocityJointLoss'),
        out_channels=3,
        type='MotionRegressionHead'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_pretrain_h36m-29ffebf5_20230719.pth',
        type='Pretrained'),
    test_cfg=dict(flip_test=True),
    type='PoseLifter')
optim_wrapper = dict(
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01))
param_scheduler = [
    dict(by_epoch=True, end=60, gamma=0.99, type='ExponentialLR'),
]
resume = False
skip_list = [
    'S9_Greet',
    'S9_SittingDown',
    'S9_Wait_1',
    'S9_Greeting',
    'S9_Waiting_1',
]
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        multiple_target=243,
        pipeline=[
            dict(
                encoder=dict(
                    concat_vis=True,
                    num_keypoints=17,
                    rootrel=True,
                    type='MotionBERTLabel'),
                type='GenerateTarget'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'factor',
                    'camera_param',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=1,
        seq_step=1,
        test_mode=True,
        type='Human36mDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        mode='mpjpe',
        skip_list=[
            'S9_Greet',
            'S9_SittingDown',
            'S9_Wait_1',
            'S9_Greeting',
            'S9_Waiting_1',
        ],
        type='MPJPE'),
    dict(
        mode='p-mpjpe',
        skip_list=[
            'S9_Greet',
            'S9_SittingDown',
            'S9_Wait_1',
            'S9_Greeting',
            'S9_Waiting_1',
        ],
        type='MPJPE'),
]
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=10)
train_codec = dict(
    concat_vis=True, mode='train', num_keypoints=17, type='MotionBERTLabel')
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        multiple_target=243,
        multiple_target_step=81,
        pipeline=[
            dict(
                encoder=dict(
                    concat_vis=True,
                    mode='train',
                    num_keypoints=17,
                    type='MotionBERTLabel'),
                type='GenerateTarget'),
            dict(
                flip_label=True,
                keypoints_flip_cfg=dict(center_mode='static', center_x=0.0),
                target_flip_cfg=dict(center_mode='static', center_x=0.0),
                type='RandomFlipAroundRoot'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'factor',
                    'camera_param',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=1,
        type='Human36mDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        encoder=dict(
            concat_vis=True,
            mode='train',
            num_keypoints=17,
            type='MotionBERTLabel'),
        type='GenerateTarget'),
    dict(
        flip_label=True,
        keypoints_flip_cfg=dict(center_mode='static', center_x=0.0),
        target_flip_cfg=dict(center_mode='static', center_x=0.0),
        type='RandomFlipAroundRoot'),
    dict(
        meta_keys=(
            'id',
            'category_id',
            'target_img_path',
            'flip_indices',
            'factor',
            'camera_param',
        ),
        type='PackPoseInputs'),
]
val_cfg = dict()
val_codec = dict(
    concat_vis=True, num_keypoints=17, rootrel=True, type='MotionBERTLabel')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        multiple_target=243,
        pipeline=[
            dict(
                encoder=dict(
                    concat_vis=True,
                    num_keypoints=17,
                    rootrel=True,
                    type='MotionBERTLabel'),
                type='GenerateTarget'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'factor',
                    'camera_param',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=1,
        seq_step=1,
        test_mode=True,
        type='Human36mDataset'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        mode='mpjpe',
        skip_list=[
            'S9_Greet',
            'S9_SittingDown',
            'S9_Wait_1',
            'S9_Greeting',
            'S9_Waiting_1',
        ],
        type='MPJPE'),
    dict(
        mode='p-mpjpe',
        skip_list=[
            'S9_Greet',
            'S9_SittingDown',
            'S9_Wait_1',
            'S9_Greeting',
            'S9_Waiting_1',
        ],
        type='MPJPE'),
]
val_pipeline = [
    dict(
        encoder=dict(
            concat_vis=True,
            num_keypoints=17,
            rootrel=True,
            type='MotionBERTLabel'),
        type='GenerateTarget'),
    dict(
        meta_keys=(
            'id',
            'category_id',
            'target_img_path',
            'flip_indices',
            'factor',
            'camera_param',
        ),
        type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Pose3dLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
