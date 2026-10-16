#!/bin/bash
set -e

cd /eos/user/p/piosifid/MyBhive

echo "Activating b_hive conda env"
source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
conda activate b_hive

export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
source setup.sh

mkdir -p logs
law run DatasetConstructorTask \
    --coffea-worker 10 \
    --config part_run3_lt \
    --dataset-version dijet_full_plus_flatpt_wide \
    --filelist filelists/dijet/train_val_samples_plus_flatpt.txt \
    > logs/dataset_dijet_full_plus_flatpt_wide_$(date +%Y%m%d_%H%M%S).log 2>&1
