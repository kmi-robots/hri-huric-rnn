import json
import matplotlib
import numpy as np
import plac
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

def read_values(folder, dataset_name):

    subfolder_list = [x for x in folder.iterdir() if x.is_dir()]
    intent_best = {}
    intent_last = {}
    slots_cond_best = {}
    slots_cond_last = {}
    for subfolder in subfolder_list:
        folder_name = subfolder.name
        file_location = subfolder / dataset_name
        try:
            with open(file_location / 'history_full.json') as f:
                content = json.load(f)
            f1_intent_scores = content['intent']['f1']
            f1_slots_cond_scores = content['slots_cond']['f1']
            f1_intent_best_value = max(f1_intent_scores)
            f1_intent_best_index = f1_intent_scores.index(f1_intent_best_value)
            f1_intent_last_value = f1_intent_scores[-1]
            f1_slots_cond_best_value = max(f1_slots_cond_scores)
            f1_slots_cond_best_index = f1_slots_cond_scores.index(f1_slots_cond_best_value)
            f1_slots_cond_last_value = f1_slots_cond_scores[-1]

            intent_best[folder_name] = f1_intent_best_value
            intent_last[folder_name] = f1_intent_last_value
            slots_cond_best[folder_name] = f1_slots_cond_best_value
            slots_cond_last[folder_name] = f1_slots_cond_last_value
        except FileNotFoundError:
            print('not found file in subfolder', subfolder)
            continue

    return {
        'intent_best': intent_best,
        'intent_last': intent_last,
        'slots_cond_best': slots_cond_best,
        'slots_cond_last': slots_cond_last
    }


def main(folder, dataset_name='huric_eb/modern'):
    main_path = Path(folder)
    aggregated = read_values(main_path, dataset_name)

    with open(main_path / 'aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    for name, values in aggregated.items():
        plt.clf()
        plt.figure(figsize=(10,10))
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


if __name__ == '__main__':
    plac.call(main)