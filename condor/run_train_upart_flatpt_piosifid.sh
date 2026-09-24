#!/bin/bash
set -e

cd /eos/user/p/piosifid/MyBhive

source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
conda activate b_hive

export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
source setup.sh

mkdir -p logs
law run TrainingTask --config part_run3_lt --dataset-version dijet_full_plus_flatpt --filelist filelists/dijet/train_val_samples_plus_flatpt.txt --training-version uparT_dijet_flatpt --model-name UParT_v0 --epochs 40 --loss-function CrossEntropyLogCosh --n-threads 8 > logs/train_uparT_flatpt_$(date +%Y%m%d_%H%M%S).log 2>&1
