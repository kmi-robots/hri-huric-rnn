import json
import matplotlib
import numpy as np
import plac
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

import re


def natural_keys(text):
    '''sorts strings with integer parts'''
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def read_values(folder, dataset_name):

    subfolder_list = sorted([x for x in folder.iterdir() if x.is_dir()], key=lambda folder: natural_keys(folder.name))
    intent_best = {}
    intent_last = {}
    slots_cond_best = {}
    slots_cond_last = {}
    bd_cond_best = {}
    bd_cond_last = {}
    ac_cond_best = {}
    ac_cond_last = {}

    intent_best_idx = {}
    slots_cond_best_idx = {}
    bd_cond_best_idx = {}
    ac_cond_best_idx = {}
    for subfolder in subfolder_list:
        folder_name = subfolder.name
        file_location = subfolder / dataset_name
        try:
            with open(file_location / 'history_full.json') as f:
                content = json.load(f)
            f1_intent_scores = content['intent']['f1']
            f1_slots_cond_scores = content['slots_cond_old']['f1']
            f1_bd_cond_scores = content['bd_cond']['f1']
            f1_ac_cond_scores = content['ac_cond_sent']['f1']
            f1_intent_best_value = max(f1_intent_scores)
            f1_intent_best_index = f1_intent_scores.index(f1_intent_best_value)
            f1_intent_last_value = f1_intent_scores[-1]
            f1_slots_cond_best_value = max(f1_slots_cond_scores)
            f1_slots_cond_best_index = f1_slots_cond_scores.index(f1_slots_cond_best_value)
            f1_slots_cond_last_value = f1_slots_cond_scores[-1]

            f1_bd_cond_best_value = max(f1_bd_cond_scores)
            f1_bd_cond_best_index = f1_bd_cond_scores.index(f1_bd_cond_best_value)
            f1_bd_cond_last_value = f1_bd_cond_scores[-1]
            f1_ac_cond_best_value = max(f1_ac_cond_scores)
            f1_ac_cond_best_index = f1_ac_cond_scores.index(f1_ac_cond_best_value)
            f1_ac_cond_last_value = f1_ac_cond_scores[-1]

            intent_best[folder_name] = f1_intent_best_value
            intent_last[folder_name] = f1_intent_last_value
            slots_cond_best[folder_name] = f1_slots_cond_best_value
            slots_cond_last[folder_name] = f1_slots_cond_last_value
            bd_cond_best[folder_name] = f1_bd_cond_best_value
            bd_cond_last[folder_name] = f1_bd_cond_last_value
            ac_cond_best[folder_name] = f1_ac_cond_best_value
            ac_cond_last[folder_name] = f1_ac_cond_last_value

            # also save the indexes
            intent_best_idx[folder_name] = f1_intent_best_index
            slots_cond_best_idx[folder_name] = f1_slots_cond_best_index
            bd_cond_best_idx[folder_name] = f1_bd_cond_best_index
            ac_cond_best_idx[folder_name] = f1_ac_cond_best_index

        except FileNotFoundError:
            print('not found file in subfolder', subfolder)
            continue

    return {
        'intent_best': intent_best,
        'intent_last': intent_last,
        'slots_cond_old_best': slots_cond_best,
        'slots_cond_old_last': slots_cond_last,
        'bd_cond_best': bd_cond_best,
        'bd_cond_last': bd_cond_last,
        'ac_cond_best': ac_cond_best,
        'ac_cond_last': ac_cond_last,
        'intent_best_idx': intent_best_idx,
        'slots_cond_best_idx': slots_cond_best_idx,
        'bd_cond_best_idx': bd_cond_best_idx,
        'ac_cond_best_idx': ac_cond_best_idx
    }



def plot_bars(main_path, aggregated):
    for name, values in aggregated.items():
        n_bars = len(values)
        height = n_bars / 3
        plt.clf()
        plt.figure(figsize=(10,height))
        labels, y = zip(*[(setup, value) for setup, value in values.items()])
        x = np.arange(len(labels))
        # vertical bars
        #plt.bar(x, y)
        #plt.xticks(x, labels)
        # horizontal bars
        plt.barh(x, y, color='lightgrey')
        plt.yticks(x, labels, size='x-small', stretch='extra-condensed', color='black', ha='left')
        plt.tick_params(axis='y', direction='out', pad=-10, right=True)

        #plt.legend(list(values.keys()), loc='lower right')
        plt.title(name)
        #plt.ylabel('f1')
        #plt.xlabel('epochs')
        plt.grid()
        file_location = main_path / name
        plt.savefig(file_location)

"""
def group_by_param_changing(aggregated):
    hyperparam_groups = {}

    for measure_name, values in aggregated.items():
        for experiment_full_name, value in values.items():
            config_name, hyper_string_all = experiment_full_name.split('___hyper:')
            hyper_param_strings = hyper_string_all.split(',')
            hyper_param_name, hyper_param_value = zip(*[s.split('=')[0], s.split('=')[1] for s in hyper_param_strings])
"""

def main(folder, dataset_name='huric/modern'):
    main_path = Path(folder)
    aggregated = read_values(main_path, dataset_name)

    with open(main_path / 'aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    plot_bars(main_path, aggregated)


if __name__ == '__main__':
    plac.call(main)
