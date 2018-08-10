#!/bin/bash
# search for good configurations, to make the choice of best architectures that will be used for hyperparam tuning

set -e

export DATASET=huric_eb/modern_right

# now evaluate both intents and slots (mm-nn#2)
# embedding size default=64
LABEL_EMB_SIZE=4 make train_joint
LABEL_EMB_SIZE=8 make train_joint
LABEL_EMB_SIZE=16 make train_joint
LABEL_EMB_SIZE=32 make train_joint
LABEL_EMB_SIZE=64 make train_joint
LABEL_EMB_SIZE=128 make train_joint
LABEL_EMB_SIZE=256 make train_joint
LABEL_EMB_SIZE=512 make train_joint
# lstm size default=100
LSTM_SIZE=4 make train_joint
LSTM_SIZE=8 make train_joint
LSTM_SIZE=16 make train_joint
LSTM_SIZE=32 make train_joint
LSTM_SIZE=64 make train_joint
LSTM_SIZE=128 make train_joint
LSTM_SIZE=256 make train_joint
LSTM_SIZE=512 make train_joint
# batch size default=16
BATCH_SIZE=2 make train_joint
BATCH_SIZE=4 make train_joint
BATCH_SIZE=8 make train_joint
BATCH_SIZE=16 make train_joint
BATCH_SIZE=32 make train_joint
BATCH_SIZE=64 make train_joint
BATCH_SIZE=128 make train_joint
BATCH_SIZE=256 make train_joint

# without attention
export ATTENTION=none
# embedding size default=64
LABEL_EMB_SIZE=4 make train_joint
LABEL_EMB_SIZE=8 make train_joint
LABEL_EMB_SIZE=16 make train_joint
LABEL_EMB_SIZE=32 make train_joint
LABEL_EMB_SIZE=64 make train_joint
LABEL_EMB_SIZE=128 make train_joint
LABEL_EMB_SIZE=256 make train_joint
LABEL_EMB_SIZE=512 make train_joint
# lstm size default=100
LSTM_SIZE=4 make train_joint
LSTM_SIZE=8 make train_joint
LSTM_SIZE=16 make train_joint
LSTM_SIZE=32 make train_joint
LSTM_SIZE=64 make train_joint
LSTM_SIZE=128 make train_joint
LSTM_SIZE=256 make train_joint
LSTM_SIZE=512 make train_joint
# batch size default=16
BATCH_SIZE=2 make train_joint
BATCH_SIZE=4 make train_joint
BATCH_SIZE=8 make train_joint
BATCH_SIZE=16 make train_joint
BATCH_SIZE=32 make train_joint
BATCH_SIZE=64 make train_joint
BATCH_SIZE=128 make train_joint
BATCH_SIZE=256 make train_joint

# configuration #5, three stages attention
export THREE_STAGES=true
export ATTENTION=slots
# embedding size default=64
LABEL_EMB_SIZE=4 make train_joint
LABEL_EMB_SIZE=8 make train_joint
LABEL_EMB_SIZE=16 make train_joint
LABEL_EMB_SIZE=32 make train_joint
LABEL_EMB_SIZE=64 make train_joint
LABEL_EMB_SIZE=128 make train_joint
LABEL_EMB_SIZE=256 make train_joint
LABEL_EMB_SIZE=512 make train_joint
# lstm size default=100
LSTM_SIZE=4 make train_joint
LSTM_SIZE=8 make train_joint
LSTM_SIZE=16 make train_joint
LSTM_SIZE=32 make train_joint
LSTM_SIZE=64 make train_joint
LSTM_SIZE=128 make train_joint
LSTM_SIZE=256 make train_joint
LSTM_SIZE=512 make train_joint
# batch size default=16
BATCH_SIZE=2 make train_joint
BATCH_SIZE=4 make train_joint
BATCH_SIZE=8 make train_joint
BATCH_SIZE=16 make train_joint
BATCH_SIZE=32 make train_joint
BATCH_SIZE=64 make train_joint
BATCH_SIZE=128 make train_joint
BATCH_SIZE=256 make train_joint


# three stages without attention
export THREE_STAGES=true
export ATTENTION=none
# embedding size default=64
LABEL_EMB_SIZE=4 make train_joint
LABEL_EMB_SIZE=8 make train_joint
LABEL_EMB_SIZE=16 make train_joint
LABEL_EMB_SIZE=32 make train_joint
LABEL_EMB_SIZE=64 make train_joint
LABEL_EMB_SIZE=128 make train_joint
LABEL_EMB_SIZE=256 make train_joint
LABEL_EMB_SIZE=512 make train_joint
# lstm size default=100
LSTM_SIZE=4 make train_joint
LSTM_SIZE=8 make train_joint
LSTM_SIZE=16 make train_joint
LSTM_SIZE=32 make train_joint
LSTM_SIZE=64 make train_joint
LSTM_SIZE=128 make train_joint
LSTM_SIZE=256 make train_joint
LSTM_SIZE=512 make train_joint
# batch size default=16
BATCH_SIZE=2 make train_joint
BATCH_SIZE=4 make train_joint
BATCH_SIZE=8 make train_joint
BATCH_SIZE=16 make train_joint
BATCH_SIZE=32 make train_joint
BATCH_SIZE=64 make train_joint
BATCH_SIZE=128 make train_joint
BATCH_SIZE=256 make train_joint