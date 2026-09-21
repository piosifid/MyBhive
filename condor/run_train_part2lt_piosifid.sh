#!/bin/bash
set -e

cd /eos/user/p/piosifid/MyBhive

source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
conda activate b_hive

export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
source setup.sh

mkdir -p logs
law run TrainingTask --config part_run3_lt --dataset-version dijet_full --training-version parT2LT_dijet --model-name ParticleTransformer2_LT --epochs 40 --n-threads 8 > logs/train_parT2LT_$(date +%Y%m%d_%H%M%S).log 2>&1
