#!/bin/bash

test_version="test_offline"

# Abort on errors
set -e
trap 'echo "An error occurred. Exiting..."; exit 1' ERR

if [ -z "$LXUSERNAME" ]; then
    echo "LXUSERNAME is not set. Please set it in your environment or export manually."
    exit 1
fi
if [ -z "$TESTDIRECTORY" ]; then
    echo "TESTDIRECTORY is not set. Please set it in your environment or export manually."
    exit 1
fi

# Ensure data directory exists
mkdir -p "$TESTDIRECTORY/data/offline"
DEST_FILE="$TESTDIRECTORY/data/offline/ntuple_merged_0.root"

# Copy or download the file
if [ ! -f $DEST_FILE ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/numpy/ntuple_merged_0.root 
    if [ -f "$SOURCE_FILE" ]; then
        echo "File exists locally."
        cp $SOURCE_FILE $DEST_FILE
    else
        echo "File not found locally. Attempting remote download..."
        scp -r $LXUSERNAME@lxplus.cern.ch:$SOURCE_FILE $DEST_FILE
        if [ $? -eq 0 ]; then
            echo "File downloaded successfully."
        else
            echo "Failed to download the file. Aborting."
            exit 1
        fi
    fi
fi

# Create filelist
FILELIST="$TESTDIRECTORY/data/offline/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

for task in TrainingTask InferenceTask ROCCurveTask; do #DatasetConstructorTask
    for config in part_run3_npy; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 1:  part_run3 + ParticleTransformer2 + epoch_lin_decay + Adam                                                       +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3_npy \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleTransformer2 \
        --TrainingTask-loss-weighting True \
        --epochs 1 \
        --batch-size 4 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer Adam \
        --n-threads 1 \
        --mixed-precision False \
        --training-precision float16 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"