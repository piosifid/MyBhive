#!/bin/bash

test_version="test_LT"

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
mkdir -p "$TESTDIRECTORY/data/LT"
DEST_FILE="$TESTDIRECTORY/data/LT/ntuple_TTHHto4B_0.root"

# Copy or download the file
if [ ! -f $DEST_FILE ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/LT/ntuple_TTHHto4B_0.root
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
FILELIST="$TESTDIRECTORY/data/LT/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do #
    for config in part_run3_lt; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 1:  part_run3_lt + ParticleTransformer2_LT + torch.compile(default) + attack (jetfool)                                    +\n"
printf "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3_lt \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleTransformer2_LT \
        --epochs 3 \
        --batch-size 4 \
        --optimizer AdamW \
        --attack jetfool \
        --attack-magnitude 0.1 \
        --use-torch-compile True \
        --torch-compile-mode default

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 2:  part_run3_lt_fp16 + ParticleTransformer2_LT_tau + torch.compile(reduce-overhead) + loss-reweighting                   +\n"
printf "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3_lt_fp16 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleTransformer2_LT_tau \
        --TrainingTask-loss-weighting True \
        --epochs 3 \
        --batch-size 4 \
        --optimizer AdamW \
        --attack-magnitude 0.1 \
        --use-torch-compile True \
        --torch-compile-mode reduce-overhead
        
printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"