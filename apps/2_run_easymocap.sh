#!/bin/bash
set -x
set -u

. $(dirname $0)/0_set_path.sh

cd $working_dir/EasyMocap

# old version
# python3 scripts/preprocess/extract_hand.py $out_dir --mode 'nanodet-resnet2d' --annot annots --video
# python3 apps/calibration/create_blank_camera.py $out_dir --shape 1080 1920
# bash ./apps/fit_run/sv1h_manol_fix.sh $out_dir $out_dir/output 

# new version
data=$out_dir
emc --data config/datasets/svimage.yml --exp config/1v1p/fixhand.yml --root ${data} --out ${data}/output --ranges 0 1800 1
