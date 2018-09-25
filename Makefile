SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

preprocess_all:
	pushd data && DATASET=huric_eb python preprocess.py && popd && ./data/huric_eb/speakers_split/group_files.py && pushd data && DATASET=huric_eb_speakers_split python preprocess.py && DATASET=framenet_subset python preprocess.py && DATASET=fate_subset python preprocess.py && popd

preprocess:
	pushd data && python preprocess.py && popd

preprocess_combinations:
	pushd data && DATASET=combinations python preprocess.py && popd

train_joint:
	python -m nlunetwork.main
