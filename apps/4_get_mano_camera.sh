#!/bin/bash
set -x
set -u

. $(dirname $0)/0_set_path.sh

cd $working_dir
python tools/preprocess_manoRT.py $out_dir --step 5