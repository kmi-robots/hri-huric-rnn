# HuRIC evaluation

Some experiments with RNN and HuRIC dataset

## Requirements

- python3-pip
- virtualenv (recommended)
- install `requirements.txt`


## Common problems

When installing the requirements, says "no space on device": this is because you may have a very small tmpfs. To fix that, edit your `/etc/fstab` with something like:

```
tmpfs     /tmp     tmpfs     defaults,size=10G,mode=1777     0     0
```

Then reboot and check with `df -h`
