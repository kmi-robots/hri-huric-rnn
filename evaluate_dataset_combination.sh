#!/bin/bash
# this script runs the model creation and evaluation for the dataset combination

set -e

export OUTPUT_FOLDER=dataset_combination
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

DATASET=huric_eb/modern_right make train_joint
DATASET=huric_eb/with_framenet make train_joint
MODE=test_all DATASET=huric_eb/modern_right make train_joint
MODE=test DATASET=huric_eb/modern_right make train_joint

# evaluation steps
python nlunetwork/results_aggregator.py nlunetwork/results/framenet/results/eval_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100 subset_right
python nlunetwork/results_aggregator.py nlunetwork/results/framenet/results/eval_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100 subset_both