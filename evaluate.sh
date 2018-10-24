#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

best_config_h=`python select_best.py nlunetwork/results/tuning/aggregated_huric.json`
best_config_fn=`python select_best.py nlunetwork/results/tuning/aggregated_fn.json`


# 1. cross validate on HuRIC with the tuned parameters from HuRIC
set -a
source nlunetwork/results/tuning/${best_config_h}/${DATASET}/config.env
set +a
export OUTPUT_FOLDER=tuning/optimized/
export MODE=cross
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}


# 2. train FrameNet and test on HuRIC (transfer learning?)
# 2.1. with the hyperparams from FrameNet
# 2.2. with the hyperparams from HuRIC

# train on full FrameNet with the hyperparam from FrameNet
set -a
source nlunetwork/results/tuning/${best_config_fn}/${DATASET}/config.env
set +a
export OUTPUT_FOLDER=tuning/optimized/
export MODE=train_all
python nlunetwork/main.py

# train all FrameNet subset on the hyperparams of HuRIC
set -a
source nlunetwork/results/tuning/${best_config_h}/${DATASET}/config.env
set +a
export DATASET=framenet/subset_both
export OUTPUT_FOLDER=tuning/optimized/
export MODE=train_all
python nlunetwork/main.py

# determine the paths of the models
best_config_fn_train_all=${best_config_fn/cross/train_all}
best_config_h_train_all=${best_config_h/cross_nested/train_all}
# TODO asggiustare sta riga che il dataset non quadra

# now test all with fn hyperparam
export MODE=test_all
export MODEL_PATH=nlunetwork/results/tuning/optimized/${best_config_fn_train_all}/framenet/subset_both/
export DATASET=huric_eb/modern_right
python nlunetwork/main.py

# now test all with h hyperparams
export MODE=test_all
export MODEL_PATH=nlunetwork/results/tuning/optimized/${best_config_h_train_all}/framenet/subset_both/
export DATASET=huric_eb/modern_right
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}