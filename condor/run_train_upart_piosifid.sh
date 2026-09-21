#!/bin/bash
set -e

cd /eos/user/p/piosifid/MyBhive

source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
conda activate b_hive

export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
source setup.sh

mkdir -p logs
law run TrainingTask --config part_run3_lt --dataset-version dijet_full --training-version uparT_dijet --model-name UParT_v0 --epochs 40 --loss-function CrossEntropyLogCosh --n-threads 8 > logs/train_uparT_$(date +%Y%m%d_%H%M%S).log 2>&1
