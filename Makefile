SHELL := /bin/bash
.PHONY: default
default:
	echo choose a target

preprocess:
	pushd data && python preprocess.py && popd

train_joint:
	python -m nlunetwork.main
