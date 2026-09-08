#!/bin/bash

# Run ROC curve
echo "Starting WorkingPointTask"
law run ROCCurveTask \
    --training-version train_balanced \
    --dataset-version train_balanced \
    --test-dataset-version summer22 \
    --config part_run3_lt \
    --model-name ParticleTransformer2_LT \
    --epochs 200 \
    --n-threads 16
    # --epochs 300
    # --epochs 150

    # --loss-function CrossEntropyLogCosh \
    # --model-name UParT_v0 \
    # --model-name ParticleTransformer2 \
    # --model-name ParticleNet_InPro \
echo "Finished WorkingPointTask"