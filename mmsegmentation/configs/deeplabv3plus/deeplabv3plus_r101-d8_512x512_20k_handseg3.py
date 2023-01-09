_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/handseg3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/finetune_20k.py'
]
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=2), 
    auxiliary_head=dict(num_classes=2),
    backbone=dict(depth=101))

load_from = 'work_dirs/deeplabv3plus_r101-d8_512x512_20k_handseg2/latest.pth'