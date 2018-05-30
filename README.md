# HuRIC evaluation

Some experiments with RNN and HuRIC dataset

## Requirements

- python3-pip
- virtualenv (recommended)
- install `requirements.txt`

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
