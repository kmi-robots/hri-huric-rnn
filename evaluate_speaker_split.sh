#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

# chosen architecture: three stages with attention
export THREE_STAGES=true
export ATTENTION=both


# this time test on folds 1-4 and test on 5
export MODE=eval

export BATCH_SIZE=2

# the chosen hyperparams
export LABEL_EMB_SIZE=64
export LSTM_SIZE=128

export MAX_EPOCHS=50

DATASET="huric_eb/speakers_split/en_nation/true" make train_joint
DATASET="huric_eb/speakers_split/en_nation/false" make train_joint
DATASET="huric_eb/speakers_split/native/solid english speaker" make train_joint
DATASET="huric_eb/speakers_split/native/weak english speaker" make train_joint
DATASET="huric_eb/speakers_split/native/yes" make train_joint
DATASET="huric_eb/speakers_split/proficiency/yes" make train_joint
DATASET="huric_eb/speakers_split/proficiency/no" make train_joint