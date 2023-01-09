#!/bin/bash
set -x
set -u

. $(dirname $0)/0_set_path.sh

mkdir -p $out_dir
mkdir -p $out_dir/video
cp $seq_dir/video.MOV $out_dir/video/video.MOV
cp $seq_dir/bg.jpg $out_dir/bg.jpg

cd $working_dir
python3 tools/extract_images.py $out_dir --image images
python3 tools/image_downsample.py $out_dir
