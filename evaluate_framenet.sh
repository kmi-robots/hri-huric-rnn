#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

export OUTPUT_FOLDER=framenet/results/
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
python nlunetwork/results_aggregator.py nlunetwork/results/framenet/results/eval_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100 subset_right
python nlunetwork/results_aggregator.py nlunetwork/results/framenet/results/eval_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100 subset_both