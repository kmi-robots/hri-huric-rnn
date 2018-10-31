#!/bin/bash
# this is the script for the hyperparameter tuning

# the possible parameter values and architecture have been chosen from the first round of architecture exploration

set -e

export DATASET=huric/modern_right
# chosen architecture: three stages with attention
export THREE_STAGES=true
export ATTENTION=slots


# hyper param tuning on folds 1-4, don't touch fold 5
export MODE=dev_cross

# smaller batches have shown better performances in the exploration of architectures
export BATCH_SIZE=2

LABEL_EMB_SIZE=16 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=32 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=64 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=16 LSTM_SIZE=128 make train_joint
LABEL_EMB_SIZE=32 LSTM_SIZE=128 make train_joint
LABEL_EMB_SIZE=64 LSTM_SIZE=128 make train_joint
LABEL_EMB_SIZE=16 LSTM_SIZE=256 make train_joint
LABEL_EMB_SIZE=32 LSTM_SIZE=256 make train_joint
LABEL_EMB_SIZE=64 LSTM_SIZE=256 make train_joint