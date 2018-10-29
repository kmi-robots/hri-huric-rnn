#!/bin/bash

set -e

export MAX_EPOCHS=20
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


# for FrameNet fulltexts
export DATASET=framenet/subset_both
export MODE=cross

for f in $CONF_FILES
do
    CONFIG_FILE=${f} python nlunetwork/main.py
done

# aggregate the epochs
python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}
mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_fn_ft.json

# for FrameNet lu
export MAX_EPOCHS=10 # more samples, less epochs
export DATASET=framenet/modern_lu_subset_right
export MODE=cross

for f in $CONF_FILES
do
    CONFIG_FILE=${f} python nlunetwork/main.py
done

# aggregate the epochs
python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}
mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_fn_lu.json


echo "Hyperparameter tuning results:"
python select_best.py nlunetwork/results/tuning/aggregated_huric.json
python select_best.py nlunetwork/results/tuning/aggregated_fn_ft.json
python select_best.py nlunetwork/results/tuning/aggregated_fn_lu.json
