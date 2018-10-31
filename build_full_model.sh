#!/bin/bash
# this script builds the three models (HuRIC, FrameNet, HuRIC+FrameNet)

set -e


export OUTPUT_FOLDER=train_all
export MODE=train_all
export CONFIG_FILE=configurations/conf_4.env

DATASET=huric/modern_right python nlunetwork/main.py
DATASET=framenet/subset_both python nlunetwork/main.py
DATASET=huric/with_framenet python nlunetwork/main.py
