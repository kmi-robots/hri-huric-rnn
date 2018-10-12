#!/bin/env/python

import os

params_list_values = [
    {
        'name': 'LABEL_EMB_SIZE',
        'values': [4,8,16,32,64,128,256,512]
    },
    {
        'name': 'LSTM_SIZE',
        'values': [4,8,16,32,64,128,256,512]
    },
    {
        'name': 'BATCH_SIZE',
        'values': [2,4,8,16]
    },
    {
        'name': 'THREE_STAGES',
        'values': ['false', 'true', 'true_highway']
    },
    {
        'name': 'ATTENTION',
        'value': ['intents', 'slots', 'both', 'none']
    }
]

little_params_list_values = [
    {
        'name': 'LABEL_EMB_SIZE',
        'values': [16,32,64]
    },
    {
        'name': 'LSTM_SIZE',
        'values': [64,128,256]
    },
    {
        'name': 'BATCH_SIZE',
        'values': [2]
    },
    {
        'name': 'THREE_STAGES',
        'values': ['true_highway']
    },
    {
        'name': 'ATTENTION',
        'values': ['both']
    }
]

MY_PATH = os.path.dirname(os.path.abspath(__file__))

def recurrent_step(params_list, index, current_params, n_created):
    if index >= len(params_list):
        # end
        string = '\n'.join(['{}={}'.format(k, v) for k, v in current_params.items()])
        with open('{}/conf_{}.env'.format(MY_PATH, n_created), 'w') as f:
            f.write(string)
        return n_created + 1
    current_choice = params_list[index]
    for v in current_choice['values']:
        current_params[current_choice['name']] = v
        n_created = recurrent_step(params_list, index + 1, current_params, n_created)
    return n_created

recurrent_step(little_params_list_values, 0, {}, 0)