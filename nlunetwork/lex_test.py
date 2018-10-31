import json
import os
import requests

import numpy as np
from dotenv import load_dotenv, find_dotenv
from aws_requests_auth.aws_auth import AWSRequestsAuth

from nlunetwork import metrics, data

load_dotenv(find_dotenv())

def load_test_set():
    with open('data/huric/modern/preprocessed/fold_5.json') as f:
        content = json.load(f)
    return content['data']


def call_lex(sentence):
    auth = AWSRequestsAuth(aws_access_key=os.environ['AWS_ACCESS_KEY'] ,
          aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
          aws_host='runtime.lex.eu-west-1.amazonaws.com',
          aws_region='eu-west-1',
          aws_service='lex')

    response = requests.post('https://runtime.lex.eu-west-1.amazonaws.com/bot/train_only/alias/latest/user/someRandomUserId/text', auth=auth, json={'inputText': sentence})
    result = response.json()
    return result

def find_slot_spans(words, lex_slots):
    result = ['O'] * len(words)
    sentence = ' '.join(words)
    w_lengths = [len(w) + 1 for w in words]
    w_start_indexes = np.cumsum([0] + w_lengths[:-1])
    w_id_by_start_idx = {start_idx: w_id for w_id, start_idx in enumerate(w_start_indexes)}
    print(w_id_by_start_idx)
    if lex_slots:
        for slot_name, value in lex_slots.items():
            if value:
                print(slot_name, value)
                start_char_idx = sentence.find(value)
                end_char_idx = start_char_idx + len(value)
                print(start_char_idx, end_char_idx)
                for start_idx, w_id in w_id_by_start_idx.items():
                    if start_idx == start_char_idx:
                        result[w_id] = 'B-{}'.format(slot_name)
                    elif start_idx > start_char_idx and start_idx < end_char_idx:
                        result[w_id] = 'I-{}'.format(slot_name)

    return result

def main():
    gold = load_test_set()
    intent_predictions = []
    slots_predictions = []
    for idx, sample in enumerate(gold):
        lex_res = call_lex(' '.join(sample['words']))
        print(lex_res)
        intent_pred = lex_res['intentName'] or 'Wrong'
        slots_iob = find_slot_spans(sample['words'], lex_res['slots'])
        print(slots_iob)
        intent_predictions.append(intent_pred)
        slots_predictions.append(slots_iob)
        #break # TODO remove me
    zero_2d = np.zeros((len(gold), 50))
    zero_3d = np.zeros((len(gold), 50, 50))
    res = metrics.clean_predictions(slots_predictions, intent_predictions, gold, zero_2d, zero_3d, zero_3d, zero_3d)
    print(res)
    data.save_predictions('lex/results/json/epoch_0', 'full', res)

if __name__ == '__main__':
    main()