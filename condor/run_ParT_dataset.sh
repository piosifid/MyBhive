#!/bin/bash

# Run dataset
echo "Starting DatasetConstructorTask"
law run DatasetConstructorTask \
    --coffea-worker 10 \
    --config part_run3_lt \
    --dataset-version summer22_2000_8000 \
    --filelist all_samples.txt
    # --dataset-version train_enhanced \
    # --chunk-size 1000000 \
    # --filelist train_val_samples_enhanced.txt
    # --dataset-version summer22 \
    # --filelist all_samples.txt
    # --dataset-version train \
    # --filelist train_val_samples.txt
    # --chunk-size 10000 \
echo "Finished DatasetConstructorTask"