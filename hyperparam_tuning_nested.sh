#!/bin/bash

set -e

export MAX_EPOCHS=1
export OUTPUT_FOLDER=tuning

CONF_FILES=configurations/conf_*.env


# for HuRIC only
export DATASET=huric_eb/modern_right
export MODE=cross_nested

for f in $CONF_FILES
do
    CONFIG_FILE=${f} python nlunetwork/main.py
done

# aggregate the epochs
python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}
mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_huric.json


# for FrameNet
export DATASET=framenet/subset_both
export MODE=cross

for f in $CONF_FILES
do
    CONFIG_FILE=${f} python nlunetwork/main.py
done

# aggregate the epochs
python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}
mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_fn.json


echo "Hyperparameter tuning results:"
python select_best.py nlunetwork/results/tuning/aggregated_huric.json
python select_best.py nlunetwork/results/tuning/aggregated_fn.json
