#!/bin/bash

test_version="test_hlt"

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

mkdir -p "$TESTDIRECTORY/data/HLT"
DEST_FILE="$TESTDIRECTORY/data/HLT/hlt_test_TT.root"

# Copy or download the file
if [ ! -f $DEST_FILE ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/hlt/out_TT_1.root
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
FILELIST="$TESTDIRECTORY/data/HLT/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

echo "Deleting old HLT tests."
for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do #DatasetConstructorTask
    for config in hlt_run3; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done

echo "Begining testing..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 1:  hlt_run3 + DeepJetHLT + epoch_lin_decay + AdamW + attacks (pgd) + torch.compile(default) + mixed precision                  +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJetHLT \
        --epochs 2 \
        --batch-size 4 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer AdamW \
        --mixed-precision True \
        --use-torch-compile True \
        --torch-compile-mode default \
        --attack pgd \
        --attack-magnitude 0.01 \
        --test-attack pgd \
        --test-attack-magnitude 0.01 


printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 2:  hlt_run3 + DeepJetTransformerHLT + batch_lin_decay + RAdam + attacks (jetfool) + mixed precision                            +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJetTransformerHLT \
        --epochs 2 \
        --batch-size 4 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer RAdam \
        --mixed-precision True \
        --attack jetfool \
        --attack-magnitude 0.3 \
        --test-attack jetfool \
        --test-attack-magnitude 0.3 


printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 3:  hlt_run3 + UParticleNet_InProHLT + batch_lin_decay + Adam + no attacks + torch.compile(reduce-overhead)                     +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name UParticleNet_InProHLT \
        --loss-function CrossEntropyLogCoshHLT \
        --epochs 2 \
        --batch-size 4 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer Adam \
        --use-torch-compile True \
        --torch-compile-mode reduce-overhead 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 4:  hlt_run3 + GlobalParticleTransformerHLT + batch_cosine_warmup + AdamW                   +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name GlobalParticleTransformerHLT \
        --epochs 2 \
        --batch-size 4 \
        --lr-scheduler batch_cosine_warmup \
        --lr-decay-factor 0.1 \
        --optimizer AdamW 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"