#!/bin/bash
# this script runs the model creation and evaluation for the dataset combination

set -e

export OUTPUT_FOLDER=dataset_combination/
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

# 80%H --> 20%H
DATASET=huric_eb/modern_right make train_joint
# 80%H + 100%FN --> 20%H
DATASET=huric_eb/with_framenet make train_joint

# some experiments with a fully FN trained model
DATASET=framenet/subset_both make train_joint
export FN_MODEL=nlunetwork/results/dataset_combination/train_all_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100/framenet/subset_both/
export OUTPUT_FOLDER=dataset_combination/fn_model
# 100%FN --> 100%H
MODE=test_all DATASET=huric_eb/modern_right MODEL_PATH=${FN_MODEL} make train_joint
# 100%FN --> 20%H
MODE=test DATASET=huric_eb/modern_right MODEL_PATH=${FN_MODEL} make train_joint

# some experiments with FATE
# TODO



# collapse histories (find max among epochs)
export COMMON_FOLDER_NAME=_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper\:LABEL_EMB_SIZE\=64\,LSTM_SIZE\=128\,BATCH_SIZE\=2\,MAX_EPOCHS\=100
python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} modern_right
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/huric_eb/modern_right/

python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} with_framenet
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/huric_eb/with_framenet/