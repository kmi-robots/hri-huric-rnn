import json

import numpy as np
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from IPython.display import HTML, display
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

from nlunetwork import data

def load_json(folder, epoch=99):
    json_location = Path(folder) / 'json' / 'epoch_{}'.format(epoch) / 'prediction_fold_full.json'
    with open(json_location) as f:
        content = json.load(f)
    return content['samples']

def get_intent_attention_arrays(sentence_data):
    """returns true (lexical units) and pred attentions"""
    lexical_units = [0] * sentence_data['length']
    for lu in sentence_data['lexical_unit_ids']:
        lexical_units[lu - sentence_data['start_token_id']] = 1
    return lexical_units, sentence_data['intent_attentions']

#def get_bd_attention_arrays(sentence_data):
#    return sentence_data['words'], sentence_data['slots_pred']
   
def get_color(value):
    """Colors a cell by the value provided [0,1] if value is number, otherwise white"""
    
    if isinstance(value, str):
        return 'rgb(255,255,255)'
        print('str', value)
    else:
        v = "%.4f" %  (255 - value * 255)
        return 'rgb({}, 255,{})'.format(v, v)
    
def float_to_str(value):
    if isinstance(value, str):
        return value
    else:
        return "%.4f" % float(value)
    
def display_sequences(row_names, sequences):
    html_str = '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td><b>{}</b></td>'.format(row_name) +
            ''.join(['<td style="background-color: {};">{}</td>'.format(get_color(value), float_to_str(value)) for value in row])
            for row_name, row in zip(row_names, sequences)
        )
    )
    display(HTML(html_str))

def display_align(row_labels, col_labels, values):
    html_str = '<table><tr>{}</tr><tr>{}</tr></table>'.format(
        # the headers
        ''.join(['<td>{}</td>'.format(t) for t in [''] + row_labels]),
        # the body
        '</tr><tr>'.join(
            '<td>{}</td>'.format(col_labels[row_idx]) +
            ''.join(['<td style="background-color: {};">{}</td>'.format(get_color(value), float_to_str(value)) for value in row])
            for row_idx, row in enumerate(values)
        )
    )
    display(HTML(html_str))

def display_all(samples, display_bd=True, display_ac=True, display_slots=True, max_show=None):
    for idx, sample in enumerate(samples):
        print('true:', sample['intent_true'], 'pred:', sample['intent_pred'])
        attentions_true, attentions_pred = get_intent_attention_arrays(sample)
        display_sequences(['words', 'lexical_unit','attention_intent', 'slots_true', 'slots_pred'],
                          [sample['words'], attentions_true, attentions_pred, sample['slots_true'], sample['slots_pred']])
        bd = data.slots_to_iob_only(sample['slots_pred'])
        ac = data.slots_to_types_only(sample['slots_pred'])
        if display_bd:
            display_align(sample['words'], bd, sample['bd_attentions'])
        bd_and_words = ['{}+{}'.format(b, w) for b, w in zip(bd, sample['words'])]
        if display_ac:
            display_align(bd_and_words, ac, sample['ac_attentions'])
        if display_slots:
            display_align(sample['words'], sample['slots_pred'], sample['slots_attentions'])
        # don't make the browser crash with all those tables
        if max_show and idx > max_show:
            return

def align_accuracy_argmax(samples):
    true_lexical_units = [s['lexical_unit_ids'][0] - 1 for s in samples]
    pred_lexical_units = [np.argmax(s['intent_attentions']) for s in samples]
    compared = [1 if t == p else 0 for t, p in zip(true_lexical_units, pred_lexical_units)]
    return sum(compared) / len(compared)

        
def align_score(samples):
    true_lexical_units = []
    for s in samples:
        true_s = [1 if idx + 1 in s['lexical_unit_ids'] else 0 for idx in range(s['length'])]
        #print('t', true_s)
        true_lexical_units.extend(true_s)
    #print(np.shape(true_lexical_units))
    
    precisions_vs_k = {}
    recalls_vs_k = {}
    f1_vs_k = {}
    
    for how_many in range(1, 10):
        pred_lexical_units = []
        for s in samples:
            padded_intent_att = np.pad(s['intent_attentions'], (0, 50 - s['length']), 'constant')
            pred_k_indexes = np.argpartition(padded_intent_att, -how_many)[-how_many:]
            #print(pred_k_indexes)
            pred_k = [1 if idx in pred_k_indexes else 0 for idx in range(s['length'])]
            #print(pred_k)
            pred_lexical_units.extend(pred_k)
        #print('true', true_lexical_units[:30])
        #print('pred', pred_lexical_units[:30])
        p, r, f1, support = precision_recall_fscore_support(true_lexical_units, pred_lexical_units, average='binary')
        #print('k=', how_many, 'p=', p, 'r=', r, 'f1=', f1, 'support=', support)
        precisions_vs_k[how_many] = p # with average='binary' the scores are for the 1 label, the interesting one
        recalls_vs_k[how_many] = r
        f1_vs_k[how_many] = f1
        
    display_alignment_vs_k(precisions_vs_k, recalls_vs_k, f1_vs_k)
    return precisions_vs_k, recalls_vs_k, f1_vs_k

def display_alignment_vs_k(precisions, recalls, f1s):
    plt.clf()
    plt.plot(precisions.keys(), precisions.values(), label='precision')
    plt.plot(recalls.keys(), recalls.values(), label='recall')
    plt.xlabel('k-top')
    plt.legend(['precision', 'recall'], loc='lower right')
    plt.show()

def group_samples_by_frame(samples):
    groups = {k: list(v) for k,v in groupby(sorted(samples, key=lambda s: s['intent_true']), key=lambda s: s['intent_true'])}
    return groups

def get_words_by_attention(samples):
    """Given a set of samples, returns a list of (word, average_attention) sorted by decreasing attention"""
    bow_cumulative = defaultdict(lambda: 0)
    for s in samples:
        for w_idx, w in enumerate(s['words']):
            bow_cumulative[w] += s['intent_attentions'][w_idx]
        #print(s['intent_attentions'])
    bow_cumulative = {k: score / len(samples) for k, score in bow_cumulative.items()}
    result = sorted([(w, w_sum) for (w, w_sum) in bow_cumulative.items()], key=lambda x: x[1], reverse=True)
    return result
