#!/bin/bash

# Run ROC curve
echo "Starting ROCCurveTask"
law run ROCCurveTask \
    --training-version train \
    --dataset-version train \
    --test-dataset-version summer22 \
    --config part_run3_lt \
    --loss-function CrossEntropyLogCosh \
    --model-name UParT_v0 \
    --epochs 200 \
    --n-threads 8
    # --epochs 300
    # --epochs 150

    # --loss-function CrossEntropyLogCosh \
    # --model-name UParT_v0 \
    # --model-name ParticleTransformer2_LT \
    # --model-name ParticleNet_InPro \
echo "Finished ROCCurveTask"