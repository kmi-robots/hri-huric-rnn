#!/bin/bash
# this is the script for the evaluation of huric

set -e

export DATASET=huric_eb/modern

# evaluate by only training the intents (mm-nn#1)
LOSS_SUM=intent make train_joint
# now evaluate both intents and slots (mm-nn#2)
make train_joint
# only IOB boundaries, without slot labels (mm-nn#3)
SLOTS_TYPE=iob_only make train_joint

# without attention
# evaluate by only training the intents (mm-nn#1)
ATTENTION=none LOSS_SUM=intent make train_joint
# now evaluate both intents and slots (mm-nn#2)
ATTENTION=none make train_joint
# only IOB boundaries, without slot labels (mm-nn#3)
ATTENTION=none SLOTS_TYPE=iob_only make train_joint

# configuration #5, three stages
#LOSS_SUM=intent make train_joint
THREE_STAGES=true make train_joint
THREE_STAGES=true ATTENTION=none make train_joint