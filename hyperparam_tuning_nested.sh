#!/bin/bash

set -e

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

best_config=`python select_best.py nlunetwork/results/tuning/aggregated.json`

set -a
source nlunetwork/results/tuning/${best_config}/${DATASET}/config.env
set +a
export OUTPUT_FOLDER=tuning/optimized/
export MODE=cross
python nlunetwork/main.py

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/optimized/ ${DATASET}