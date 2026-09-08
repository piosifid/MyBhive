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
    SOURCE_FILE=/eos/cms/store/group/phys_btag/b-hive/test_files/data/offline/ntuple_merged_0.root
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

for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do #
    for config in part_run3 part_fp16_run3 offline_run3 UParT_v0_run3; do
        path="$DATA_PATH/$task/$config/debug/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 1:  part_run3 + ParticleNet_InPro + epoch_lin_decay + Adam                                                       +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleNet_InPro \
        --epochs 1 \
        --batch-size 4 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer Adam \
        --mixed-precision False

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 2:  part_fp16_run3 + DeepJet + batch_cosine_warmup + AdamW + attacks (pgd)                                       +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_fp16_run3 \
        --training-precision float16 \
        --testing-precision float16 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJet \
        --epochs 2 \
        --batch-size 4 \
        --lr-scheduler batch_cosine_warmup \
        --lr-decay-factor 0.1 \
        --optimizer AdamW \
        --attack pgd \
        --attack-magnitude 0.1

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 3:  part_run3 + DeepJetTransformer + batch_cosine_warmup + AdamW + attacks (pgd)                                 +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJetTransformer \
        --TrainingTask-loss-weighting True \
        --epochs 1 \
        --batch-size 4 \
        --optimizer AdamW \
        --attack pgd \
        --verbose True \
        --attack-magnitude 0.1

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 4:  offline_run3 + ParticleTransformer + epoch_lin_decay + RAdam + attack (jetfool) + test_attack (optimizer) + torch.compile(default) + mixed-precision    +\n"
printf "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config offline_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleTransformer \
        --epochs 1 \
        --batch-size 4 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer RAdam \
        --attack jetfool \
        --attack-magnitude 0.1 \
        --test-attack optimizer \
        --test-attack-magnitude 0.15 \
        --use-torch-compile True \
        --torch-compile-mode default \
        --mixed-precision True

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 5:  UParT_v0_run3 + UParT_v0 + batch_lin_decay + AdamW + mixed-precision  --  attacks are not yet implemented for UParT     +\n"
printf "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config UParT_v0_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name UParT_v0 \
        --loss-function CrossEntropyLogCosh \
        --epochs 1 \
        --batch-size 4 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer AdamW \
        --mixed-precision True

printf "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+         Test 6:  part_run3 + ParticleTransformer2 + epoch_lin_decay + AdamW + torch.compile(reduce-overhead) + mixed-precision           +\n"
printf "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --debug True \
        --DatasetConstructorTask-chunk-size 1000 \
        --config part_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleTransformer2 \
        --epochs 3 \
        --batch-size 4 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer AdamW \
        --mixed-precision True \
        --use-torch-compile True \
        --torch-compile-mode reduce-overhead 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"