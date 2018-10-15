#!/bin/bash

set -e


# for HuRIC only
export DATASET=huric_eb/modern_right

export OUTPUT_FOLDER=tuning/
export MODE=cross_nested
export MAX_EPOCHS=50

CONF_FILES=configurations/conf_*.env

for f in $CONF_FILES
do
    set -a
    source $f
    set +a
    python nlunetwork/main.py
done

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}

mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_huric.json
best_config=`python select_best.py nlunetwork/results/tuning/aggregated_huric.json`

set -a
source nlunetwork/results/tuning/${best_config}/${DATASET}/config.env
set +a
export OUTPUT_FOLDER=tuning/optimized/
export MODE=cross
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}

# for FrameNet
export DATASET=framenet/subset_both

export OUTPUT_FOLDER=tuning/
export MODE=cross
export MAX_EPOCHS=50

CONF_FILES=configurations/conf_*.env

for f in $CONF_FILES
do
    set -a
    source $f
    set +a
    python nlunetwork/main.py
done

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ ${DATASET}

mv nlunetwork/results/tuning/aggregated.json nlunetwork/results/tuning/aggregated_fn.json
best_config=`python select_best.py nlunetwork/results/tuning/aggregated_fn.json`

set -a
source nlunetwork/results/tuning/${best_config}/${DATASET}/config.env
set +a
export OUTPUT_FOLDER=tuning/optimized/
export MODE=test_all
export MODEL_PATH=nlunetwork/results/tuning/${best_config}/${DATASET}
export DATASET=huric_eb/modern
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}