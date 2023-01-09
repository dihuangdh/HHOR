_base_ = './deeplabv3plus_r101-d8_512x512_160k_handseg2.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
