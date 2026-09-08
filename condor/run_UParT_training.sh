#!/bin/bash
set -e

# cp -r "/afs/cern.ch/user/f/fopanagi/private/b-hive" .
cd ..

# Load modules and activate environment
export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output"
source setup.sh
eval "$(conda shell.bash hook)"
conda activate b-hive

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
echo "Starting training..."
law run TrainingTask \
    --training-version train_UParT \
    --dataset-version train_UParT \
    --config UParT_highPT \
    --model-name UParT_v0 \
    --epochs 2 \
    2>&1 | tee logs/training_UParT_progress_$(date +%Y%m%d_%H%M%S).log
