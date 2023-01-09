#!/bin/bash
set -x
set -u

. $(dirname $0)/0_set_path.sh

cd $working_dir
python tools/preprocess_backgrounds.py $out_dir

cd $working_dir/BackgroundMattingV2
for file in $out_dir/images/*; do
    # echo $(basename ${file})
    python inference_images.py --images-src $out_dir/images/$(basename ${file}) \
        --images-bgr $out_dir/backgrounds/$(basename ${file}) \
        --model-type mattingrefine --model-backbone resnet101 \
        --model-checkpoint pytorch_resnet101.pth \
        --output-dir $out_dir/masks/$(basename ${file}) \
        --output-types pha \
        --model-refine-mode thresholding
done

cd $working_dir
python tools/preprocess_mask.py $out_dir