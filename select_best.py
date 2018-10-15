#!/usr/bin/env python

"""Shows which is the best configuration from an `aggregated.json` file"""

import os
import json
import plac
from collections import defaultdict


def main(aggregated_location):
    with open(aggregated_location) as f:
        content = json.load(f)

    # the weights for comparing the different fitness measures
    weights = {
        'intent_best': 1,
        'bd_cond_best': 1,
        'ac_cond_best': 1
    }

    results = defaultdict(lambda: 0)

    for k, values in content.items():
        if k in weights:
            for conf_name, conf_v in values.items():
                results[conf_name] += weights[k] * conf_v

    sorted_result = sorted([(k,v) for k, v in results.items()], key=lambda el: el[1], reverse=True)

    print(sorted_result[0][0])
    return sorted_result

if __name__ == '__main__':
    plac.call(main)