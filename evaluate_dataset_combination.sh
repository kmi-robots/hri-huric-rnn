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

# cross fold HuRIC
MODE=cross DATASET=huric/modern_right make train_joint
# 80%H --> 20%H
DATASET=huric/modern_right make train_joint
# 80%H + 100%FN_subset --> 20%H
DATASET=huric/with_framenet make train_joint
DATASET=huric/with_fate make train_joint
DATASET=huric/with_framenet_and_fate make train_joint
DATASET=fate/with_framenet make train_joint
# 80%FN_subset --> 20%FN_subset
DATASET=framenet/subset_both make train_joint
# 80%FA_subset --> 20%FA_subset
# train all FN_subset
MODE=train_all DATASET=framenet/subset_both make train_joint
# train all FA_subset
MODE=train_all DATASET=fate/subset_both make train_joint

# experiment also with full FrameNet and full FATE
DATASET=framenet/modern_both make train_joint
DATASET=fate/modern_both make train_joint

# some experiments with a fully FN trained model
export FN_MODEL=nlunetwork/results/dataset_combination/train_all_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100/framenet/subset_both/
export OUTPUT_FOLDER=dataset_combination/fn_model/
# 100%FN_subset --> 100%H
MODE=test_all DATASET=huric/modern_right MODEL_PATH=${FN_MODEL} make train_joint
# 100%FN_subset --> 20%H
MODE=test DATASET=huric/modern_right MODEL_PATH=${FN_MODEL} make train_joint

# some experiments with FATE
export FATE_MODEL=nlunetwork/results/dataset_combination/train_all_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper:LABEL_EMB_SIZE=64,LSTM_SIZE=128,BATCH_SIZE=2,MAX_EPOCHS=100/fate/subset_both/
export OUTPUT_FOLDER=dataset_combination/fate_model/
# 100%FA_subset --> 100%H
MODE=test_all DATASET=huric/modern_right MODEL_PATH=${FATE_MODEL} make train_joint
# 100%FA_subset --> 20%H
MODE=test DATASET=huric/modern_right MODEL_PATH=${FATE_MODEL} make train_joint



# collapse histories (find max among epochs)
export COMMON_FOLDER_NAME=_loss_both_slottype_full_we_large_recurrent_cell_lstm_attention_both_three_stages_true_highway___hyper\:LABEL_EMB_SIZE\=64\,LSTM_SIZE\=128\,BATCH_SIZE\=2\,MAX_EPOCHS\=100

python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} modern_right
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated_modern_righ.json

# also the cross validation
python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/cross${COMMON_FOLDER_NAME} modern_right
mv nlunetwork/results/dataset_combination/cross${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/cross${COMMON_FOLDER_NAME}/aggregated_modern_righ.json

# huric with framenet, test 20%
python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} with_framenet
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated_with_framenet.json

# this one will capture fate and framenet
python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} modern_both
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated_modern_both.json

# with_fate and with_framenet_and_fate
python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} with_fate
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated_with_fate.json

python nlunetwork/results_aggregator.py nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME} with_framenet_and_fate
mv nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated.json nlunetwork/results/dataset_combination/eval${COMMON_FOLDER_NAME}/aggregated_with_framenet_and_fate.json