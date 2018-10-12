#!/bin/bash

set -e


export OUTPUT_FOLDER=tuning/
export MODE=cross_nested
export MAX_EPOCHS=50

CONF_FILES=configurations/conf_*.env

for f in $CONF_FILES
do
    set -o allexport
    source $f
    set -o allexport
    python nlunetwork/main.py
done

python nlunetwork/results_aggregator.py nlunetwork/results/tuning/ huric_eb/modern_right
