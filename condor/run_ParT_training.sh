#!/bin/bash

# Run training
echo "Starting TrainingTask"
law run TrainingTask \
    --training-version train_enhanced \
    --dataset-version train_enhanced \
    --config part_run3_lt \
    --model-name ParticleTransformer2_LT \
    --epochs 40 \
    --n-threads 64 \
    --resume-training True
    # --loss-weighting True #used only for balanced versions
    # --lr-scheduler epoch_lin_decay \
    # --learning-rate 0.00001 \
    # --lr-decay-factor 0.1 \
    # --resume-training True
    # --loss-weighting True

    # --batch-size 512 \
    # --epochs 300 \
    # --lr-scheduler batch_cosine_warmup
    # --epochs 150 \
    # --lr-scheduler batch_cosine_warmup
    # --learning-rate 0.00001 \
    # --lr-decay-factor 0.00001

    # --attack pgd --attack-magnitude 0.2 \
    # --loss-function CrossEntropyLogCosh \
    # --model-name UParT_v0 \
    # --model-name ParticleTransformer2_LT \
    # --model-name ParticleNet_InPro \
echo "Finished TrainingTask"