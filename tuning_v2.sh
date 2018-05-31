#!/bin/bash
# this is the script for the evaluation of huric

# the parameters have been chosen from the first round of tuning

set -e

export DATASET=huric_eb/modern
export BATCH_SIZE=2

## config 1
#export ATTENTION=none
#
#LABEL_EMB_SIZE=16 LSTM_SIZE=256 make train_joint
#LABEL_EMB_SIZE=32 LSTM_SIZE=256 make train_joint
#LABEL_EMB_SIZE=64 LSTM_SIZE=256 make train_joint
#
#
## config 2, two stages attention
#export ATTENTION=slots
#
#LABEL_EMB_SIZE=16 LSTM_SIZE=256 make train_joint
#LABEL_EMB_SIZE=32 LSTM_SIZE=256 make train_joint
#LABEL_EMB_SIZE=64 LSTM_SIZE=256 make train_joint
#
#
## config 3, three stages without attention
#export THREE_STAGES=true
#export ATTENTION=none
#
#LABEL_EMB_SIZE=16 LSTM_SIZE=64 make train_joint
#LABEL_EMB_SIZE=32 LSTM_SIZE=64 make train_joint
#LABEL_EMB_SIZE=64 LSTM_SIZE=64 make train_joint
#
# config 4, three stages with attention
export THREE_STAGES=true
export ATTENTION=slots

LABEL_EMB_SIZE=16 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=32 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=64 LSTM_SIZE=64 make train_joint
LABEL_EMB_SIZE=16 LSTM_SIZE=128 make train_joint
LABEL_EMB_SIZE=32 LSTM_SIZE=128 make train_joint
LABEL_EMB_SIZE=64 LSTM_SIZE=128 make train_joint