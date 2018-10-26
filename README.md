# HuRIC evaluation

Some experiments with RNN and HuRIC dataset

## Requirements

Python 3.6 is required, because python2 has not been tested and tensorflow does not support python3.7.

- python3-pip: use your package manager e.g. `apt`
- virtualenv (recommended): to use python 3.6 do `virtualenv venv --python=python3.6`
- install the dependencies: `pip install -r requirements.txt`
- for running the preprocessing, you need the spacy model with dependency parsing: `spacy download en`

## Scripts

- [first_experiments.sh](first_experiment.sh) contains some first experiment with the dataset and the the network
- [architectures_exploration.sh](architectures_exploration.sh) is for choosing which variation of the network (2 vs 3 stages, attention vs not attention)
- [hyperparam_tuning.sh](hyperparam_tuning.sh) script for hyperparam tuning
- [evaluate.sh](evaluate.sh) script for evaluation

## Transfer the results

```bash
# collapse
huric_rnn/joint$ python results_aggregator.py results
# copy to your machine
huric_rnn/joint$ rsync -zarvP --prune-empty-dirs --include "*/" --include "*.png" --include "aggregated.json" --include "history_full.json" --exclude "*" martino.mensio@$IP:/home/martino.mensio/huric_rnn/joint/results results/google_cloud
```
where `IP` env is set correctly

## Common problems

When installing the requirements, says "no space on device": this is because you may have a very small tmpfs. To fix that, edit your `/etc/fstab` with something like:

```
tmpfs     /tmp     tmpfs     defaults,size=10G,mode=1777     0     0
```

Then reboot and check with `df -h`

## Amazon lex test

Take [this file](data/huric_eb/modern/amazon/lexTrainBot.json.zip), that contains the 4 train folds, upload to amazon lex console, build the model and set an alias.

Then create an `.env` in the root folder file with content:
```
AWS_ACCESS_KEY=PUT_THERE_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=PUT_THERE_SECRET_ACCESS_KEY
```

Then execute:

```bash
python -m joint.lex_test
python -m joint.evaluate_predictions_stored lex/results
```

And look at the results in `lex/results`.

## Run local server

1. Build the models: `./build_full_model.sh`
2. Run the server `FLASK_APP=server.py flask run` (optionally add `--port PORT_NUMBER` to use another port)
3. go to [localhost:5000/nlu?text=sentence](http://localhost:5000/nlu?text=sentence)

(To test different models use the env variable `MODEL_PATH` like `MODEL_PATH=nlunetwork/results/train_all/conf_4/huric_eb/modern_right/ FLASK_APP=server.py flask run`)

## Notebooks

In the folder [notebooks](notebooks) there are jupyter notebooks for the analysis of results.
