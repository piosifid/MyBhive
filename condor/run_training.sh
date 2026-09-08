#!/bin/bash
set -e

echo "Copying b-hive folder"
cp -r "/afs/cern.ch/user/f/fopanagi/public/Workspace/b-hive" .
echo "cd inside the b-hive folder"
cd b-hive
# cd ..
ls -lh

# Load modules and activate environment
echo "Activating the b-hive conda env"
eval "$(conda shell.bash hook)"
conda activate b-hive
echo "Setting up the b-hive env with setup.sh"
export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/full"
# export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/dijet"
# export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/test"
# export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/with_cut"
# export DATA_PATH="/eos/cms/store/group/phys_exotica/dijet/Dijet13TeV/fopanagi/b-hive/output/without_cut"
source setup.sh

./condor/run_ParT_training.sh > $DATA_PATH"/logs/train_ParT_progress_"$(date +%Y%m%d_%H%M%S)".log" 2>&1