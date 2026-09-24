#!/bin/bash
set -e

cd /eos/user/p/piosifid/MyBhive

source /eos/user/u/uwillems/miniforge3/etc/profile.d/conda.sh
conda activate b_hive

export DATA_PATH="/eos/user/p/piosifid/b-hive-output/"
source setup.sh

mkdir -p logs
law run WorkingPointTask --config part_run3_lt --dataset-version dijet_full --training-version particleNet_dijet --test-dataset-version dijet_test --test-filelist filelists/test/all_samples.txt --model-name ParticleNet_InPro --epochs 40 --n-threads 8 > logs/wp_particleNet_$(date +%Y%m%d_%H%M%S).log 2>&1
