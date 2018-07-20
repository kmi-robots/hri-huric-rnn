#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

# preprocessing steps
DATASET=framenet_subset_right make preprocess
DATASET=framenet_subset_both make preprocess

# chosen architecture: three stages with attention
export THREE_STAGES=true_highway
export ATTENTION=both


# this time test on folds 1-4 and test on 5
export MODE=eval

export BATCH_SIZE=2

# the chosen hyperparams
export LABEL_EMB_SIZE=64
export LSTM_SIZE=128
#export MAX_EPOCHS=100
export DATASET=framenet/subset_right
make train_joint

export DATASET=framenet/subset_both
make train_joint

# evaluation steps
