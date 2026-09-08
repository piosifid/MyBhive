#!/bin/bash

# Run training
echo "Starting training..."
law run TrainingTask \
    --training-version train_ParticleNet \
    --dataset-version train_ParT \
    --config ParT_highPT \
    --model-name ParticleNet_InPro \
    --epochs 500
echo "Finished training"