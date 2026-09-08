#!/bin/bash

test_version="test_paired"

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
mkdir -p "$TESTDIRECTORY/data/PAIReD_LL_CC_BB"
DEST_FILE="$TESTDIRECTORY/data/PAIReD_LL_CC_BB/paired_test.root"

# Copy or download the file
if [ ! -f $TESTDIRECTORY/data/PAIReD_LL_CC_BB/paired_test.root ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/paired/paired_test.root
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
FILELIST="$TESTDIRECTORY/data/PAIReD_LL_CC_BB/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do #
    for config in PAIReD_ParT_cls; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."


printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 1:  PAIReD_ParT_cls + PAIReDTagger + epoch_lin_decay + AdamW + attack (pgd)       +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config PAIReD_ParT_cls \
        --debug True \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --DatasetConstructorTask-chunk-size 30000 \
        --DatasetConstructorTask-coffea-worker 16 \
        --test-filelist $FILELIST  \
        --model-name LZ4PAIReDTagger \
        --epochs 3 \
        --batch-size 512 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer AdamW \
        --attack pgd \
        --attack-magnitude 0.02 \
        --attack-iterations 2 
        
printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"