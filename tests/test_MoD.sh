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
mkdir -p "$TESTDIRECTORY"
mkdir -p "$TESTDIRECTORY/data"
mkdir -p "$TESTDIRECTORY/data/MoD"
DEST_FILE="$TESTDIRECTORY/data/MoD/ntuple_merged_0.root"

# Copy or download the file
if [ ! -f $DEST_FILE ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/MoD/ntuple_merged_0.root
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
FILELIST="$TESTDIRECTORY/data/MoD/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do #
    for config in mod_offline_run3; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 1:  mod_offline_run3 + MoDJet + batch_lin_decay + AdamW + attack (pgd) + test_attack (jetfool)                   +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --config mod_offline_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --DatasetConstructorTask-chunk-size 10000 \
        --model-name MoDJet \
        --epochs 2 \
        --batch-size 512 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.01 \
        --attack pgd \
        --attack-magnitude 0.05 \
        --test-attack jetfool \
        --test-attack-magnitude 0.05 \
        --optimizer AdamW \
        --betas 0.9,0.999 
        
printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"