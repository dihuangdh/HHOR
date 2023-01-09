#!/bin/bash
set -x
set -u

. $(dirname $0)/0_set_path.sh

cd $working_dir/mmsegmentation
python demo/preprocess_grasphand_semantic.py $out_dir configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_handseg3.py \
    work_dirs/deeplabv3plus_r101-d8_512x512_160k_handseg3/latest.pth --device cuda:0