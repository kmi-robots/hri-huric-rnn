import json

import numpy as np
from collections import defaultdict
from itertools import groupby
from pathlib import Path
import xml.etree.ElementTree as ET
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

def load_xmls(folder):
    path = Path(folder)
    files_list = [el for el in path.iterdir() if el.is_file()]
    file_contents = []
    for file in sorted(files_list):
        with open(file) as file_in:
            tree = ET.parse(file_in)
        root = tree.getroot()
        file_contents.append(root)
    return file_contents

def get_lu_pos(root, verbose=False):
    """Returns the POS for all the LU in the current document"""
    w_id_to_pos = {t.attrib['id']: t.attrib['pos'] for t in root.findall('tokens/token')}
    #print(w_id_to_pos)
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    #print(lu_idxs)
    lu_pos = [w_id_to_pos[id] for id in lu_idxs]
    if verbose:
        print(root.attrib['id'], lu_pos)
    return lu_pos

def get_lu_are_roots(root, verbose=False):
    """Returns whether the lexicalUnits are the roots in dependencies"""
    roots_id = [d.attrib['to'] for d in root.findall('dependencies/dep') if d.attrib['type'] == 'root']
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_are_roots = [str(l in roots_id) for l in lu_idxs]
    if verbose:
        print(root.attrib['id'], lu_are_roots)
    return lu_are_roots

def get_lengths(root, verbose=False):
    """Returns the length of the command"""
    # TODO shouldn't be frame-based? find min and max token id and do difference
    return [len(root.findall('tokens/token'))]

def get_lu_depths(root, verbose=False):
    """Returns the depths in the dependency tree of the lexicalUnits"""
    # TODO the depth should be relative to the frame
    edges = [(d.attrib['from'], d.attrib['to'], d.attrib['type']) for d in root.findall('dependencies/dep')]
    #print(edges)
    to_father = {e[1]: e[0] for e in edges}
    depths = {}
    for el, f in to_father.items():
        current_el = el
        depth = 0
        # the root has id == '0', the second condition is only to avoid infinite looping
        while f != '0' and depth < len(edges):
            depth += 1
            current_el = f
            f = to_father[current_el]
        if f != '0':
            # broken annotations
            depth = -1
        depths[el] = depth
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_depths = [depths[l] for l in lu_idxs]
    if verbose:
        print(root.attrib['id'], lu_depths)
    return lu_depths

def get_lu_positions(root, verbose=False):
    """Get the position of lexicalUnits in the command"""
    # TODO the position should be relative to the frame
    lu_idxs = [lu.attrib['id'] for lu in root.findall('semantics/frameSemantics/frame/lexicalUnit/token')]
    lu_positions = [int(l) for l in lu_idxs]
    return lu_positions

def get_corpus_complexity_statistics(dataset_location):
    """Loads the dataset in xml format HuRIC 1.2 and computes some measures"""
    xml_docs = load_xmls(dataset_location)
    lu_pos_all = defaultdict(lambda: 0)
    lu_are_roots_all = defaultdict(lambda: 0)
    lengths_all = defaultdict(lambda: 0)
    lu_depths_all = defaultdict(lambda: 0)
    lu_positions_all = defaultdict(lambda: 0)
    for doc in xml_docs:
        lu_pos = get_lu_pos(doc)
        for p in lu_pos:
            lu_pos_all[p] += 1
        lu_are_roots = get_lu_are_roots(doc)
        for r in lu_are_roots:
            lu_are_roots_all[r] += 1
        lengths = get_lengths(doc)
        for l in lengths:
            lengths_all[l] += 1
        lu_depths = get_lu_depths(doc)
        for d in lu_depths:
            lu_depths_all[d] += 1
        lu_positions = get_lu_positions(doc)
        for p in lu_positions:
            lu_positions_all[p] += 1
    return {
        'lu_pos': lu_pos_all,
        'lu_are_roots': lu_are_roots_all,
        'lengths': lengths_all,
        'lu_depths': lu_depths_all,
        'lu_positions': lu_positions_all
    }

def plot_measures(datasets_dict, show=True, path=None):
    """Given a dict {conf_name: measures} produces a set of plots, one for each measure, with all the configurations inside"""
    # reverse the nested dict to {measure_name: {conf_name: {label: value}}
    stats_by_measure = defaultdict(dict)
    for conf_name, measures in datasets_dict.items():
        for measure_name, measure in measures.items():
            stats_by_measure[measure_name][conf_name] = measure
    # adjust the width of the columns
    width = 0.35 / len(datasets_dict.keys())
    #print(stats_by_measure)
    for measure_name, confs_measures in stats_by_measure.items():
        fig, ax = plt.subplots()
        # merge all the measures indexes, that can have different values among the configurations
        # first get the set of all unique indexes
        all_indexes = sorted(set([m for measure in confs_measures.values() for m in measure.keys()]))
        #print(all_indexes)
        # then build a map {index_value: position_in_all_indexes}
        all_indexes = {val: idx for idx, val in enumerate(all_indexes)}
        #print(all_indexes)
        bars = []
        for idx, (conf_name, measure) in enumerate(confs_measures.items()):
            corresponding_x_indexes = [all_indexes[m] for m in measure]
            bar = ax.bar(np.array(corresponding_x_indexes) + (idx * width), [m for m in measure.values()], width)
            bars.append(bar)
        ax.set_ylabel('counts')
        ax.set_title(measure_name)
        ax.set_xticks(np.arange(len(all_indexes)) + (width / 2 * (len(datasets_dict) - 1)))
        ax.set_xticklabels(list(all_indexes.keys()))
        ax.legend(bars, confs_measures.keys())
        if path:
            _ = plt.savefig('{}_{}.png'.format(path, measure_name))
        if show:
            plt.show()
        plt.close(fig)