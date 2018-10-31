#!/bin/bash
# this is the script for final evaluation

# the architecture was chosen in architecture exploration
# the hyperparam are chosen in hyperparameter tuning

set -e

best_config_h=`python select_best.py nlunetwork/results/tuning/aggregated_huric.json`
#best_config_fn=`python select_best.py nlunetwork/results/tuning/aggregated_fn.json`


# 1. cross validate on HuRIC with the tuned parameters from HuRIC
# select the parameters from the best configuration (don't trust the conf file name, maybe some custom parameter changed)
set -a
source nlunetwork/results/tuning/${best_config_h}/huric/modern_right/config.env
set +a
export DATASET=huric/modern_right
export OUTPUT_FOLDER=tuning/optimized/h_only
export MODE=cross
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/h_only ${DATASET}


# 2. train FrameNet and test on HuRIC (transfer learning?)
# 2a. FrameNet fullText (fewer samples)
# 2b. FrameNet lu (lots of samples)
# 2.1. with the hyperparams from FrameNet
# 2.2. with the hyperparams from HuRIC

# train on full FrameNet with the hyperparam of FrameNet
set -a
source nlunetwork/results/tuning/${best_config_fn}/framenet/subset_both/config.env
set +a
export DATASET=framenet/subset_both
export OUTPUT_FOLDER=tuning/optimized/fn_only_hyper_fn
export MODE=train_all
python nlunetwork/main.py


# framenet from lu
MAX_EPOCHS=10 DATASET=framenet/modern_lu_subset_right OUTPUT_FOLDER=tuning/experiment_lu MODE=train_all CONFIG_FILE=configurations/conf_4.env python nlunetwork/main.py
DATASET=huric/modern_right OUTPUT_FOLDER=tuning/experiment_lu MODE=test_all MODEL_PATH=nlunetwork/results/tuning/experiment_lu/conf_4/framenet/modern_lu_subset_right/ python nlunetwork/main.py

MAX_EPOCHS=10 DATASET=framenet/modern_lu_subset_right OUTPUT_FOLDER=tuning/experiment_lu_cross MODE=cross CONFIG_FILE=configurations/conf_4.env python nlunetwork/main.py


# train all FrameNet subset on the hyperparam of HuRIC
set -a
source nlunetwork/results/tuning/${best_config_h}/huric/modern_right/config.env
set +a
export DATASET=framenet/subset_both
export OUTPUT_FOLDER=tuning/optimized/fn_only_hyper_h
export MODE=train_all
python nlunetwork/main.py


# now test all with fn hyperparam
export MODE=test_all
export OUTPUT_FOLDER=tuning/optimized/fn_only_hyper_fn
export MODEL_PATH=nlunetwork/results/tuning/optimized/fn_only_hyper_fn/${best_config_fn}/framenet/subset_both/
export DATASET=huric/modern_right
python nlunetwork/main.py

# now test all with h hyperparams
export MODE=test_all
export OUTPUT_FOLDER=tuning/optimized/fn_only_hyper_h
export MODEL_PATH=nlunetwork/results/tuning/optimized/fn_only_hyper_h/${best_config_h}/framenet/subset_both/
export DATASET=huric/modern_right
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}


# 3. dataset combination

# simpler: conf_4
CONFIG_FILE=configurations/conf_4.env MAX_EPOCHS=10 DATASET=huric/with_framenet OUTPUT_FOLDER=tuning/both_experiment MODE=cross python nlunetwork/main.py
CONFIG_FILE=configurations/conf_4.env MAX_EPOCHS=10 DATASET=huric/with_framenet_lu OUTPUT_FOLDER=tuning/both_experiment MODE=cross python nlunetwork/main.py

# cross-validation with the hyperparam from FrameNet
set -a
source nlunetwork/results/tuning/${best_config_fn}/${DATASET}/config.env
set +a
export DATASET=huric/with_framenet
export OUTPUT_FOLDER=tuning/optimized/
export MODE=cross
python nlunetwork/main.py

# cross-validation with the hyperparams of HuRIC
set -a
source nlunetwork/results/tuning/${best_config_h}/huric/modern_right/config.env
set +a
export DATASET=huric/with_framenet
export OUTPUT_FOLDER=tuning/optimized/
export MODE=cross
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}