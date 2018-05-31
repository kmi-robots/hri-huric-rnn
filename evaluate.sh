#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

export DATASET=huric_eb/modern
# chosen architecture: three stages with attention
export THREE_STAGES=true
export ATTENTION=slots


# hyper param tuning on folds 1-4, don't touch fold 5
export MODE=eval

export BATCH_SIZE=2

# the chosen hyperparams
LABEL_EMB_SIZE=TODO
LSTM_SIZE=TODO
make train_joint