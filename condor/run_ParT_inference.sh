#!/bin/bash

# Run inference
echo "Starting InferenceTask"
law run InferenceTask \
    --training-version train_ParT \
    --dataset-version train_ParT \
    --test-dataset-version test_ParT \
    --config ParT_highPT \
    --model-name ParticleTransformer \
    --epochs 300
    # --epochs 150
echo "Finished InferenceTask"