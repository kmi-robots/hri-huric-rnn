import json
import os
import plac
import numpy as np

from collections import defaultdict
from pathlib import Path

from . import metrics

# TODO this code is mostly copy-paste from main, do refactor

def save_file(file_content, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open('{}/{}'.format(file_path, file_name) , 'w') as out_file:
        json.dump(file_content, out_file, indent=2, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(path_str):
    result_path = Path(path_str)
    # TODO for subfolder
    json_results_folder = result_path / 'json'
    epoch_dirs = [el for el in json_results_folder.iterdir() if el.is_dir()]
    history = defaultdict(lambda:  defaultdict(create_empty_array))
    create_empty_array = lambda: np.zeros((len(epoch_dirs), ))
    for epoch_dir in epoch_dirs:
        epoch_n = int(epoch_dir.name.split('_')[1])
        with open(epoch_dir / 'prediction_fold_full.json') as f:
            epoch_data = json.load(f)['samples']
        #print(epoch_data)
        epoch_perf = metrics.evaluate_epoch(epoch_data)
        #print(epoch_perf)
        save_file(epoch_perf, '{}/scores'.format(path_str), 'epoch_{}.json'.format(epoch_n))
        for key, measures in epoch_perf.items():
            if isinstance(measures, dict):
                for measure_name, value in measures.items():
                    history[key][measure_name][epoch_n] = value

    to_plot_precision = {output_type: values['precision'] for output_type, values in history.items()}
    to_plot_recall = {output_type: values['recall'] for output_type, values in history.items()}
    to_plot_f1 = {output_type: values['f1'] for output_type, values in history.items()}
    metrics.plot_history('{}/f1.png'.format(path_str) , to_plot_f1)
    save_file(history, path_str, 'history_full.json')



if __name__ == '__main__':
    plac.call(main)